%matplotlib inline

import numpy as np

import pandas as pd

import datetime as dt

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid

from os import listdir, makedirs

from tqdm import tqdm



from sklearn.model_selection import train_test_split



from keras.models import Sequential

from keras import optimizers

from keras.layers import Dense, Activation,Dropout, Conv2D,Flatten,BatchNormalization,MaxPooling2D

from keras.utils import to_categorical
Path = '../input/'

print(listdir(f'{Path}'))
train = pd.read_csv(f'{Path}/train.csv')
train.head()
plt.imshow(np.array(train.drop(['label'],axis =1).iloc[3]).reshape(28,28))
test = pd.read_csv(f'{Path}/test.csv');test.head()
train, valid = train_test_split(train, test_size=0.2)
train.head()
valid.head()
y = to_categorical(train.label)

valid_y = to_categorical(valid.label)

y
train.drop(['label'],inplace = True,axis = 1)

valid.drop(['label'],inplace = True,axis =1)
#Normalize 

train = train/255

valid = valid/255

test = test/255
train.head()
#Dense Neural Network

model = Sequential()

model.add(Dense(1024, input_shape=(784,)))

model.add(Activation('relu'))

model.add(Dropout(0.1))



model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.1))



model.add(Dense(256))

model.add(Activation('relu'))

model.add(Dropout(0.1))

model.add(Dense(10, activation='softmax'))

model.summary()

sgd = optimizers.SGD(lr=0.001, momentum=0.00, decay=0.00, nesterov=False)

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])
def fit(epoch = 10):

    model.fit(x=train, y=y, batch_size=64, epochs=epoch, verbose=1, callbacks=None, validation_data=(valid,valid_y), shuffle=False, class_weight=None, sample_weight=None, initial_epoch=0)
fit(10)
score_train = model.evaluate(train, y, verbose=0)

score_valid = model.evaluate(valid, valid_y, verbose=0)

print('Train Score : {}  Validation Score  :  {}'.format(score_train,score_valid))
fit(5)
fit(5)
train_cnn = np.array(train).reshape(-1,28,28,1)

valid_cnn = np.array(valid).reshape(-1,28,28,1)

test_cnn = np.array(test).reshape(-1,28,28,1)
fig, ax = plt.subplots(1,2)

ax[0].imshow(np.array(train)[0].reshape(28,28))

ax[1].imshow(train_cnn[0].reshape(28,28))
model_cnn = Sequential()



model_cnn.add(Conv2D(32, (3, 3), input_shape=(28,28,1)))

model_cnn.add(BatchNormalization(axis=-1))

model_cnn.add(Activation('relu'))

model_cnn.add(Conv2D(32, (3, 3)))

model_cnn.add(BatchNormalization(axis=-1))

model_cnn.add(Activation('relu'))

model_cnn.add(MaxPooling2D(pool_size=(2,2)))



model_cnn.add(Conv2D(64,(3, 3)))

model_cnn.add(BatchNormalization(axis=-1))

model_cnn.add(Activation('relu'))

model_cnn.add(Conv2D(64, (3, 3)))

model_cnn.add(BatchNormalization(axis=-1))

model_cnn.add(Activation('relu'))

model_cnn.add(MaxPooling2D(pool_size=(2,2)))



model_cnn.add(Flatten())

model_cnn.add(Dense(512,activation='relu'))



model_cnn.add(Dense(10,activation='softmax'))

model_cnn.summary()
sgd = optimizers.SGD(lr=0.01, momentum=0.00, decay=0.00, nesterov=False)

model_cnn.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['acc'])
def fit_cnn(epoch = 10):

    model_cnn.fit(x=train_cnn, y=y, batch_size=64, epochs=epoch, verbose=1, callbacks=None, validation_data=(valid_cnn,valid_y), shuffle=False, class_weight=None, sample_weight=None, initial_epoch=0)
fit_cnn(10)