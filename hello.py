# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 00:58:22 2024

@author: kim
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

f=open("BTC_USD_2019-02-28_2020-02-27-CoinDesk.csv", 'r', encoding='utf-8')
coindesk_data=pd.read_csv(f, header=0)

seq=coindesk_data[['Closing Price (USD)']].to_numpy()
print('데이터 길이:', len(seq), '\n앞쪽 5개 값:', seq[0:5])

#시계열 데이터를 윈도우 단위로 자르는 함수
def seq2dataset(seq,window,horizon):
    X=[]; Y=[]
    for i in range(len(seq)-(window+horizon)+1):
        x=seq[i:(i+window)]
        y=(seq[i+window+horizon-1])
        X.append(x); Y.append(y)
    return np.array(X), np.array(Y)

w=7
h=1

X,Y = seq2dataset(seq,w,h)
print(X.shape,Y.shape)
print(X[0], Y[0]); print(X[-1],Y[-1])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#훈련 집합과 테스트 집합으로 분할
split=int(len(X)*0.7)
x_train=X[0:split]; y_train=Y[0:split]
x_test=X[split:]; y_test=Y[split:]

# LSTM 모델 설계와 학습
model=Sequential()
model.add(LSTM(units=128, activation='relu', input_shape=x_train[0].shape))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam', metrics=['mae'])
hist=model.fit(x_train, y_train, epochs=200, batch_size=1, 
               validation_data=(x_test, y_test), verbose=2)