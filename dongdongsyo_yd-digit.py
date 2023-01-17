#-*- coding: utf-8 -*-



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import os

import tensorflow as tf



import keras

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import ModelCheckpoint,EarlyStopping

# seed 값 설정

#실행할때 마다 같은 결과를 출력하기 위한 seed값 설정

seed = 0

np.random.seed(1212)

#판다스로 data 읽어오기

df = pd.read_csv('../input/digit-recognizer/train.csv')
df.head()
df.shape
#features와 lable 분리

df_features = df.iloc[:, 1:785]

df_label = df.iloc[:, 0]
df_features

df_label

# train / test 데이터 추출

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_label, 

                                                test_size = 0.2,

                                                random_state = 1212)
# 배열 생성

X_train_arr = np.array(X_train)

X_test_arr = np.array(X_test)

Y_train_arr = np.array(Y_train)

Y_test_arr = np.array(Y_test)

# Feature Normalization 

X_train_reshape = X_train_arr.reshape(X_train_arr.shape[0], 784).astype('float32') / 255

X_test_reshape = X_test_arr.reshape(X_test_arr.shape[0], 784).astype('float32') / 255

# Feature Normalization 

Y_train_reshape = np_utils.to_categorical(Y_train_arr, 10)

Y_test_reshape = np_utils.to_categorical(Y_test_arr, 10)

X_train_reshape

Y_train_reshape

# 모델 프레임 설정

model = Sequential()

model.add(Dense(512, input_dim=784, activation='relu'))

model.add(Dense(10, activation='sigmoid'))

# 모델 실행 환경 설정

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

# 모델 최적화 설정

MODEL_DIR = './model2/'

if not os.path.exists(MODEL_DIR):

    os.mkdir(MODEL_DIR)



modelpath="./model2/{epoch:02d}-{val_loss:.4f}.hdf5"

checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# 모델의 실행

history = model.fit(X_train_reshape, Y_train_reshape, validation_data=(X_test_reshape, Y_test_reshape), epochs=100, batch_size=256, verbose=0, callbacks=[early_stopping_callback,checkpointer])

# 테스트 정확도 출력

print("\n Test Accuracy: %.4f" % (model.evaluate(X_test_reshape, Y_test_reshape)[1]))

# 그래프로 출력

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss")

legend = ax[0].legend(loc='best')



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best')
