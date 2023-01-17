import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam

import numpy as np

from random import randint

from sklearn.preprocessing import MinMaxScaler

train_sample=[]

train_label=[]

for i in range(1000):

    younger_age=randint(13,64)

    train_sample.append(younger_age)

    train_label.append(0)

    older_age=randint(65,100)

    train_sample.append(older_age)

    train_label.append(1)
train_sample=np.array(train_sample)

train_label=np.array(train_label)
scaler= MinMaxScaler(feature_range=(0, 1))

scaler_train_sample=scaler.fit_transform(train_sample.reshape(-1,1))
model = Sequential()

model.add(Dense(16, activation='relu',input_dim=1))

model.add(Dense(32, activation='relu'))

model.add(Dense(2,activation='softmax'))
model.summary()

model.compile(Adam(0.001),

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.fit(train_sample,train_label,epochs=10, batch_size=10)
test_sample=[]

test_label=[]

for i in range(500):

    younger_age=randint(13,64)

    test_sample.append(younger_age)

    test_label.append(0)

    older_age=randint(65,100)

    test_sample.append(older_age)

    test_label.append(1)

test_sample=np.array(test_sample)

test_label=np.array(test_label)
test_sample_output=model.predict_classes(test_sample, batch_size=10)
from sklearn.metrics import classification_report, confusion_matrix
OutputPredictedByModel=confusion_matrix(test_label,test_sample_output)

OutputPredictedByModel