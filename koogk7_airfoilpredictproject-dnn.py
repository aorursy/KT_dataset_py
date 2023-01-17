# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import sklearn

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# df = pd.read_csv('../input/con_5.csv', header = None)

df = pd.read_csv('../input/control-point11/Con_11.csv', header = None)



cp = 11



X_data = df.iloc[:,0:cp*2+1]



lift = df.iloc[:,(cp*2)+1].reset_index(drop=True)

drag = df.iloc[:,(cp*2)+2].reset_index(drop=True)

Y_data = lift/drag

Y_data = Y_data.astype(int)





#-------Test data 분리 -------------------

# train_x = X_data[:4200].reset_index(drop=True)

# train_y = Y_data[:4200].reset_index(drop=True)

# test_x = X_data[4200:].reset_index(drop=True)

# test_y = Y_data[4200:].reset_index(drop=True)

train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(X_data, Y_data, test_size = 0.25)

train_x = train_x.as_matrix()

train_y = train_y.as_matrix()

test_x = test_x.as_matrix()

test_y = test_y.reset_index(drop=True).as_matrix()



# train_x.describe()

# print(len(train))

# test_y.describe()

# Y_data.plot()

# Y_data = Y_data.as_matrix()

# X_data = X_data.as_matrix()

# Any results you write to the current directory are saved as output.
from keras.models import Sequential, load_model

from keras import metrics

from keras.layers import Dense, BatchNormalization

from keras.callbacks import ModelCheckpoint

import tensorflow as tf

import datetime



layers_node = [500,250,125,70]

model = Sequential()

model.add(Dense(layers_node[0], input_dim=len(train_x[0]),activation = 'relu',kernel_initializer='he_normal'))

# BatchNormalization(axis = 1)

for i in range(0,len(layers_node)-1):

    model.add(Dense(layers_node[i+1], input_dim=layers_node[i], activation='relu'))

    BatchNormalization(axis = 1)

model.add(Dense(1))





def plot_loss(history):

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Model Loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc=0)

 

def plot_acc(history):

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.title('Model accuracy')

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc=0)



#Compile Model

model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])

    

#Learning

hist = model.fit(train_x,train_y, validation_split=0.1,epochs=250,batch_size=50);





# 6. 모델 평가하기

loss_and_metrics = model.evaluate(test_x, test_y, batch_size=50)

print('## evaluation loss and_metrics ##')

print(loss_and_metrics)

plot_loss(hist)

plt.show()

plot_acc(hist)

plt.show()

#Save

from keras.models import load_model

model.save('A_dnn.h5')
from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot



%matplotlib inline



#모델 아키텍쳐 확인

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))



#모델 불러오기

model = load_model('A_dnn.h5') 

yhat = model.predict(test_x) # 모델 예측값

test_y = pd.DataFrame(test_y) # 실제 정답

yhat = pd.DataFrame(yhat)

loss = yhat- test_y

out_y = pd.concat([yhat, test_y, yhat-test_y], axis=1)

# loss.describe()

#시각화

# plt.plot(test_y)

# plt.plot(yhat)

# plt.title('Model Loss')

# plt.ylabel('Y_Value')

# plt.xlabel('Index')

# plt.legend(['Answer', 'Predict'], loc=0)

# print(loss.mean())



#정규분포 그래프

data = np.random.normal(loss.mean(), loss.std(),loss.count())

plt.hist(data, bins = 500)

plt.show()



#오차 수치

# print(out_y)