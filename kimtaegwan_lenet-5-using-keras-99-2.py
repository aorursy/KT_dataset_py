import gc

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



# 교차검증 lib

from sklearn.model_selection import StratifiedKFold,train_test_split

from tqdm import tqdm_notebook



#모델 lib

from keras.datasets import mnist

from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, Model

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, AveragePooling2D

from keras import layers

from keras.optimizers import Adam,RMSprop



#모델

from keras.applications import VGG16, VGG19, resnet50



#경고메세지 무시

import warnings

warnings.filterwarnings(action='ignore')
import os

os.listdir('../input/digit-recognizer')

datapath = '../input/digit-recognizer'
train =pd.read_csv(datapath+'/train.csv')

print(train.shape)

train.head()
test =pd.read_csv(datapath+'/test.csv')

print(test.shape)

test.head()
train_labels = train['label']

train = (train.iloc[:,1:].values).astype('float32')

test = test.values.astype('float32')
#Visualizing the data

sample = train[10, :].reshape(28,28)

plt.imshow(sample, cmap='gray')

plt.show()

print('label : ', train_labels[10])
train = train.reshape(42000, 28, 28, 1)

test= test.reshape(28000, 28, 28, 1)

# change shape using pad

train = np.pad(train, ((0,0),(2,2),(2,2),(0,0)), 'constant')

test = np.pad(test, ((0,0),(2,2),(2,2),(0,0)), 'constant')



print('train shape : ', train.shape)

print('test shape : ', test.shape)
# int64 -> float32 ,  scaling

train = train.astype('float32')/255

test = test.astype('float32')/255

X_train, X_val, y_train, y_val = train_test_split(train, train_labels, test_size=0.20, random_state=42)



#One-hot encoding the labels

print('X_train shape : ', X_train.shape)

print('X_val shape : ', X_val.shape)

print('y_train : ', y_train.shape)

print('y_val : ', y_val.shape)

y_train = to_categorical(y_train)

y_val = to_categorical(y_val)

print('y_train_to_categorical : ', y_train.shape)

print('y_val_to_categorical : ', y_val.shape)
#lenet-5 model

model = Sequential()

#Conv layer 1

model.add(layers.Conv2D(filters=6, kernel_size=(5, 5),strides=1, activation='relu', input_shape=(32,32,1)))

#Pooling layer 1

model.add(AveragePooling2D(pool_size = 2, strides = 2))

#Conv Layer2

model.add(layers.Conv2D(filters=16, kernel_size=(5, 5),strides=1, activation='relu'))

#Pooling layer 2

model.add(AveragePooling2D(pool_size = 2, strides = 2))

model.add(layers.Flatten())

#FC Layer 3

model.add(layers.Dense(120, activation='relu'))

#FC Layer 4

model.add(layers.Dense(84, activation='relu'))

#FC Layer 5

model.add(layers.Dense(10, activation = 'softmax'))





# compile

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()
datagen = ImageDataGenerator(

        rotation_range=10,  

        zoom_range = 0.10,  

        width_shift_range=0.1, 

        height_shift_range=0.1)
patient = 4

callbacks_list = [

    ReduceLROnPlateau(

        monitor = 'val_loss', 

        #Reduces learning rate to half

        factor = 0.5, 

        #위와 동일

        patience = patient / 2, 

        #min Reduces learning

        min_lr=0.00001,

        verbose=1,

        mode='min'

    )]
%%time

epochs =30

batch_size = 64

history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,y_val),

                              steps_per_epoch=X_train.shape[0] // batch_size

                              ,callbacks=callbacks_list,verbose = 1)

#predict

submission =pd.read_csv(datapath+'/sample_submission.csv')

pred = model.predict(test)

pred = np.argmax(pred,axis = 1)

submission['Label'] = pred

submission.to_csv('submission.csv',index=False)

submission.head()
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



epochs = range(len(acc))



plt.plot(epochs, acc, label='Training acc')

plt.plot(epochs, val_acc, label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.ylim(0.9,1)

plt.show()
loss = history.history['loss']

val_loss = history.history['val_loss']



plt.plot(epochs, loss, label='Training loss')

plt.plot(epochs, val_loss, label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.ylim(0,0.5)

plt.show()