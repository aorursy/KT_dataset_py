import pandas as pd

import numpy as np

import tensorflow as tf

import tensorflow.keras as kr

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
X = train.drop('label',axis=1)

y = train['label']
import matplotlib.pyplot as plt
image0 = X.values[0].reshape(28,28)
print(plt.imshow(image0))

print('lable:', y[0])
model = kr.Sequential()
model.add(kr.layers.Dense(units=28*28,input_shape=(28*28,),activation='relu'))

model.add(kr.layers.Dense(units=200,activation='relu'))

model.add(kr.layers.Dense(units=50,activation='relu'))

model.add(kr.layers.Dropout(rate=0.4))

model.add(kr.layers.Dense(units=10,activation='softmax'))
model.summary()
from sklearn.preprocessing import StandardScaler

stdscaler = StandardScaler()
X_scaled = stdscaler.fit_transform(X)

test_scaled = stdscaler.fit_transform(test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=101)
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.fit(X_train, y_train,epochs=100,batch_size=200, validation_data=(X_test,y_test))
model.evaluate(X_test, y_test)
pre = model.predict(test_scaled)
pre_class = np.argmax(pre,axis=1)
pre_class
sample_submit = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
sample_submit['Label'] = pre_class
sample_submit.to_csv('submit_simple_ANN.csv',index=False)
model_cnn = kr.Sequential()

model_cnn.add(kr.layers.Conv2D(kernel_size=5,input_shape=(28,28,1),strides=2,filters=1,activation='relu'))

model_cnn.add(kr.layers.Conv2D(kernel_size=3,strides=1,filters=6,activation='relu'))

model_cnn.add(kr.layers.MaxPool2D(3,1))

model_cnn.add(kr.layers.Flatten())

model_cnn.add(kr.layers.Dropout(rate=0.2))

model_cnn.add(kr.layers.Dense(units=50,activation='relu'))

model_cnn.add(kr.layers.Dense(units=10,activation='softmax'))
model.summary()
X_train_2D , X_test_2D , test_2D = X_train.reshape(-1,28,28,1),X_test.reshape(-1,28,28,1),test_scaled.reshape(-1,28,28,1)
model_cnn.compile(optimizer='adam',

                 loss='sparse_categorical_crossentropy',

                 metrics=['accuracy'])
model_cnn.fit(X_train_2D, y_train, epochs=200, batch_size=500, validation_data=(X_test_2D, y_test))
pre_cnn = model_cnn.predict(test_2D)
pre_cnn_class = np.argmax(pre_cnn,axis=1)
sample_submit['Label'] = pre_cnn_class
sample_submit.to_csv('submit_demo_cnn.csv',index=False)
X_train_RNN, X_test_RNN, test_RNN = X_train.reshape(-1,28,28),X_test.reshape(-1,28,28),test_scaled.reshape(-1,28,28)
X_train_RNN.shape
model_rnn = kr.Sequential()

model_rnn.add(kr.layers.GRU(units=150,input_shape=(X_train_RNN.shape[1:]),activation='relu',return_sequences=True))

model_rnn.add(kr.layers.GRU(units=50,activation='relu',return_sequences=True))

model_rnn.add(kr.layers.GRU(units=10,activation='relu',return_sequences=True))

model_rnn.add(kr.layers.Flatten())

model_rnn.add(kr.layers.Dropout(rate=0.4))

model_rnn.add(kr.layers.Dense(units=100,activation='relu'))

model_rnn.add(kr.layers.Dense(units=10,activation='softmax'))

model_rnn.summary()
model_rnn.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model_rnn.fit(X_train_RNN, y_train, epochs=15,batch_size=300, validation_data=(X_test_RNN, y_test))
pre_rnn = model_rnn.predict(test_RNN)
pre_rnn_class = np.argmax(pre_rnn, axis=1)
sample_submit['Label'] = pre_rnn_class
sample_submit.to_csv('submit_rnn.csv',index=False)