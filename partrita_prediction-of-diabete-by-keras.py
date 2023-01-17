# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# 필요한 모듈을 불러들입니다.
from keras.models import Sequential
from keras.layers import Dense
import numpy
import matplotlib.pyplot as plt
%matplotlib inline 

# random seed for reproducibility
numpy.random.seed(2018)

# 데이터 세트를 불러옵니다. 
dataset = numpy.loadtxt('../input/diabetes.csv', delimiter=",",skiprows=1) # skip the column index
# 데이터세트를 두 가지 원인(X) 과 결과(Y)로 나누어 줍니다. 
X = dataset[:,0:8]
Y = dataset[:,8]
print(X)
# create model, add dense layers one by one specifying activation function
model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu')) # input layer requires input_dim param
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # sigmoid instead of relu for final probability between 0 and 1

# compile the model, adam gradient descent (optimized)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

# call the function to fit to the data (training the network)
# verbose=0 는 진행상황을 숨깁니다.
history = model.fit(X, Y, epochs = 600, batch_size=10, verbose=0)

# 모델의 정확도를 계산합니다.
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# Get the figure and the axes
fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2, sharey=False, figsize=(10, 5))

# 모델의 정확도를 그립니다.
ax0.plot(history.history['acc']) 
ax0.set(title='model accuracy', xlabel='epoch', ylabel='accuracy')

# 모델의 오차를 그립니다.
ax1.plot(history.history['loss'])
ax1.set(title='model loss', xlabel='epoch', ylabel='loss')
# 가상의 환자 데이터 입력
patient_1 = numpy.array([[0,137,90,35,168,43.1,2.288,33]])

# 모델로 예측하기
prediction = model.predict(patient_1)

# 예측결과 출력하기
print("당뇨병에 걸릴 확률은 {}% 입니다.".format(prediction*100))