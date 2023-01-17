import numpy as np

import pandas as pd

import os

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt



plt.style.use('seaborn-whitegrid')

%matplotlib inline



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.callbacks import EarlyStopping



from keras.layers import Conv2D, MaxPool2D, Flatten

from keras.layers import DepthwiseConv2D



print(os.listdir("../input"))
df_train = pd.read_csv('../input/fashion-mnist_train.csv')

df_test = pd.read_csv('../input/fashion-mnist_test.csv')
df_train.info()
df_train.head()
y_train = df_train['label']

X_train = df_train.drop(columns=['label'])

y_test = df_test['label']

X_test = df_test.drop(columns=['label'])
## Normalize

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
## One hot encode

y_train = pd.get_dummies(y_train)

y_test = pd.get_dummies(y_test)
## Basic info about the data

print("X_train: ", X_train.shape)

print("y_train: ", y_train.shape)

print("X_test: ", X_test.shape)

print("y_test: ", y_test.shape)
def model():

    model = Sequential()

    model.add(Dense(784, activation='relu', input_dim=784))

    model.add(Dense(300, activation='relu'))

    model.add(Dense(100, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    return model
model = model()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
earlyStopper = EarlyStopping(monitor='acc', patience=1)
my_hist = model.fit(x=X_train, y=y_train, batch_size=100, epochs=100, callbacks=[earlyStopper])
eval = model.evaluate(x=X_test, y=y_test, batch_size=100)
eval
plt.plot(my_hist.history['acc'])

plt.plot(my_hist.history['loss'])

plt.legend(['accuracy', 'loss'], loc='right')

plt.title('accuracy and loss')

plt.xlabel('epoch')

plt.ylabel('accuracy/loss')

plt.show()
X_train = X_train.reshape((X_train.shape[0], 28,28,1))

X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
def modelCNN():

    model1 = Sequential()

    model1.add(Conv2D(5, kernel_size=[3,3], padding='valid', input_shape=(28,28,1)))

    model1.add(Conv2D(25, kernel_size=[5,5], padding='valid', activation='relu'))

    model1.add(MaxPool2D(pool_size=[3,3]))

    model1.add(Conv2D(50, kernel_size=[3,3], padding='same', activation='relu'))

#     model1.add(MaxPool2D(pool_size=[3,3]))

    model1.add(Conv2D(100, kernel_size=[3,3], padding='valid', activation='relu'))

    model1.add(Flatten())

    model1.add(Dense(1024, activation='relu'))

    model1.add(Dense(512, activation='relu'))

    model1.add(Dense(10, activation='softmax'))

    return model1
model1 = modelCNN()
model1.summary()
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model1.fit(x=X_train, y=y_train, batch_size=100, epochs=100, callbacks=[earlyStopper])
plt.plot(hist.history['acc'])

plt.plot(hist.history['loss'])

plt.legend(['accuracy', 'loss'], loc='right')

plt.title('accuracy and loss')

plt.xlabel('epoch')

plt.ylabel('accuracy/loss')

plt.show()
def modelDepthWise():

    model2 = Sequential()

    model2.add(DepthwiseConv2D(kernel_size=[3,3], padding='valid', depth_multiplier=5, input_shape=(28,28,1)))

    model2.add(DepthwiseConv2D(kernel_size=[5,5], padding='valid', depth_multiplier=5, activation='relu'))

    model2.add(MaxPool2D(pool_size=[3,3]))

    model2.add(DepthwiseConv2D(kernel_size=[3,3], padding='same', depth_multiplier=2, activation='relu'))

    model2.add(DepthwiseConv2D(kernel_size=[3,3], padding='valid', depth_multiplier=2, activation='relu'))

    model2.add(Flatten())

    model2.add(Dense(1024, activation='relu'))

    model2.add(Dense(512, activation='relu'))

    model2.add(Dense(10, activation='softmax'))

    return model2
modelDepthwise = modelDepthWise()
modelDepthwise.summary()
modelDepthwise.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = modelDepthwise.fit(x=X_train, y=y_train, batch_size=100, epochs=100, callbacks=[earlyStopper])
plt.plot(hist.history['acc'])

plt.plot(hist.history['loss'])

plt.legend(['accuracy', 'loss'], loc='right')

plt.title('accuracy and loss')

plt.xlabel('epoch')

plt.ylabel('accuracy/loss')

plt.show()