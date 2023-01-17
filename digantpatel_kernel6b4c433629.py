# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf



import keras

from keras.datasets import mnist

from keras import backend as K

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from keras import layers, callbacks

from keras.layers import AveragePooling2D,Add, concatenate, Convolution2D, Conv2D, Input, MaxPooling2D, UpSampling2D, Activation, ZeroPadding2D,Dropout,Conv2DTranspose

from keras.layers.core import Reshape, Permute, Flatten

from keras.layers.normalization import BatchNormalization 

from keras.layers import LeakyReLU, Dense, MaxPool2D

from keras.optimizers import RMSprop,SGD,Adam

from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.models import Sequential

from sklearn import metrics

from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from sklearn.utils import shuffle
batch_size = 500

num_classes = 10

epochs = 300

img_rows, img_cols = 28,28





# In[3]:





test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

Dig_MNIST = pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv")



x_dig=Dig_MNIST.drop('label',axis=1).iloc[:,:].values

x_dig = x_dig.reshape(x_dig.shape[0], 28, 28,1)

y_dig=Dig_MNIST.label



X=train.iloc[:,1:].values 

Y=train.iloc[:,0].values



X = X.reshape(X.shape[0], 28, 28,1) 

Y = keras.utils.to_categorical(Y, 10) 



x_test=test.drop('id', axis=1).iloc[:,:].values

x_test = x_test.reshape(x_test.shape[0], 28, 28,1)
# y_dig = keras.utils.to_categorical(y_dig, 10) 

# y_dig.shape
# X = np.concatenate((X,x_dig), axis=0)

# Y = np.concatenate((Y,y_dig), axis=0)
X,Y = shuffle(X,Y)

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size = 0.2, random_state=42)





train_datagen = ImageDataGenerator(rescale = 1./255.,

                                   rotation_range = 12,

                                   width_shift_range = 0.25,

                                   height_shift_range = 0.25,

                                   #shear_range = 0.15,

                                   zoom_range = 0.15,

                                   horizontal_flip = False,

                                   vertical_flip = False)



valid_datagen = ImageDataGenerator(rescale=1./255) 
initial_learningrate = 5e-3

batch_size = 128

epochs = 100



def lr_decay(epoch):#lrv

    return initial_learningrate * 0.99 ** epoch
learning_rate_reduction = ReduceLROnPlateau(monitor='loss', 

                                            patience=4, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
def get_model(num_classes, input_shape):

    model = Sequential()



    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Dropout(0.2))





    model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Dropout(0.2))





    model.add(Flatten())

    model.add(Dense(256, activation = "relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))

    model.add(Dense(84, activation = "relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))

    model.add(Dense(10, activation = "softmax"))

    

    return model
def lenet():

    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)))

    model.add(MaxPool2D(strides=2))

    model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))

    model.add(MaxPool2D(strides=2))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))

    model.add(Dense(84, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    

    return model
input_shape = (28,28,1)

model = get_model(num_classes, input_shape)

# model = lenet()

# optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

optimizer = Adam(lr=initial_learningrate)

# model.summary()



model.compile(loss="categorical_crossentropy",

              optimizer= optimizer,

              metrics=['accuracy'])
history = model.fit_generator(

      train_datagen.flow(X_train,Y_train, batch_size=batch_size),

      steps_per_epoch=X_train.shape[0]//batch_size,

      epochs=70,

      callbacks=[learning_rate_reduction           

               ],

      validation_data=valid_datagen.flow(X_valid,Y_valid),

      #validation_steps=50,  

      verbose=2

    ,class_weight = {0:2,1:1,2:1,3:1,4:1,5:1,6:2,7:2,8:1,9:1})
# model.load_weights('./model.h5',overwrite=True)

# preds_dig=model.predict(x_dig/255.)

# preds_dig = np.argmax(preds_dig,axis=-1)

# print(metrics.accuracy_score(preds_dig, y_dig))
predictions = model.predict(x_test/255.)

# predictions = np.argmax(predictions)
predictions.shape

# Y.shape,y2_train.shape
x2_train, x2_test, y2_train, y2_test = train_test_split(x_test, predictions, test_size=0.15)

x_train_final = np.concatenate((X,x2_train), axis=0)

y_train_final = np.concatenate((Y,y2_train), axis=0)
X_train, X_test, Y_train, Y_test = train_test_split(x_train_final, y_train_final, test_size=0.2)
keras.backend.clear_session()
model2 = get_model(num_classes, input_shape)
learning_rate_reduction2 = ReduceLROnPlateau(monitor='loss', patience=4, verbose=1, factor=0.5, min_lr=0.00001)

checkpoint2 = ModelCheckpoint("bestmodel.model", monitor='loss', verbose=1, save_best_only=True)

model2.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.002), metrics=['acc'])
history2 = model2.fit_generator(

    train_datagen.flow(X_train, Y_train, batch_size=128),

    steps_per_epoch=X_train.shape[0] // 128,

    epochs=70,

    validation_data=valid_datagen.flow(X_test, Y_test),

    validation_steps=X_test.shape[0] // 128,

    callbacks=[checkpoint2, learning_rate_reduction2],

    class_weight = {0:2,1:1,2:1,3:1,4:1,5:1,6:2,7:2,8:1,9:1})
model2.load_weights('bestmodel.model')
preds_dig=model2.predict(x_dig/255.)

preds_dig = np.argmax(preds_dig,axis=-1)

print(metrics.accuracy_score(preds_dig, y_dig))



results=model2.predict(x_test/255.0)

predictions=np.argmax(results, axis=-1)

submission = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')

submission['label'] = predictions

submission.to_csv("submission.csv",index=False)

print('Done')
misclassified = []

for i in range(len(preds_dig)):

    if preds_dig[i] != y_dig[i]:

        misclassified.append((preds_dig[i],y_dig[i]))
cnt = {}

for a,b in misclassified:

    if b not in cnt:

        cnt[b] = 0

    cnt[b] += 1
cnt
# from sklearn.metrics import plot_confusion_matrix

# import matplotlib.pyplot as plt
# plot_confusion_matrix(model, x_dig, y_dig,

#                                  display_labels=[0,1,2,3,4,5,6,7,8,9],

#                                  cmap=plt.cm.Blues,

#                                  normalize=False)