# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

from keras import backend as K

import tensorflow as tf

from tqdm import tqdm

import cv2, numpy as np

from PIL import Image

import matplotlib.pyplot as plt

from matplotlib import style

from random import shuffle

import random as rn

from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split



from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.layers import Conv2D, BatchNormalization

from keras.layers import Dropout, Flatten,Activation

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator



from keras import regularizers



print(os.listdir("../input/btds/BTDS"))



# Any results you write to the current directory are saved as output.
X=[]

Z=[]

IMG_SIZE=150

train_icecek_DIR ='../input/btds/BTDS/icecek'

train_yiyecek_DIR ='../input/btds/BTDS/yiyecek'
def assign_label(img,train_type):

    return train_type
def make_train_data(train_type,DIR):

    for img in tqdm(os.listdir(DIR)):

        label=assign_label(img,train_type)

        path = os.path.join(DIR,img)

        img = cv2.imread(path,cv2.IMREAD_COLOR)

        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

        

        X.append(np.array(img))

        Z.append(str(label))
make_train_data('icecek',train_icecek_DIR)

len(X)
make_train_data('yiyecek',train_yiyecek_DIR)

len(Z)
fig,ax=plt.subplots(2,2)

fig.set_size_inches(15,15)

for i in range(2):

    for j in range (2):

        l=rn.randint(0,len(Z))

        ax[i,j].imshow(X[l])

        ax[i,j].set_title('Tip: '+Z[l])

        

plt.tight_layout()
le=LabelEncoder()

Y=le.fit_transform(Z)

Y=to_categorical(Y,2)

X=np.array(X)

X=X/255
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.40,random_state=42)
np.random.seed(42)

rn.seed(42)

tf.set_random_seed(42)
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (6,6),padding = 'Same',activation ='relu', input_shape = (150,150,3)))

model.add(MaxPooling2D(pool_size=(2,2)))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

 



model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



model.add(Flatten())

model.add(Dense(512 ,kernel_regularizer=regularizers.l2(0.01)))

model.add(Dropout(0.5))

model.add(Activation('relu'))

model.add(Dense(2, activation = "softmax"))
batch_size=15

epochs=10

red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.3,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.3,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(x_train)
model.compile(optimizer=Adam(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
History = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_test,y_test),

                              verbose = 1, steps_per_epoch=x_train.shape[0])
loss = History.history['loss']

val_loss = History.history['val_loss']



epochs = range(1, len(loss) + 1)



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
loss = History.history['acc']

val_loss = History.history['val_acc']



epochs = range(1, len(loss) + 1)



plt.plot(epochs, loss, 'bo', label='Training acc')

plt.plot(epochs, val_loss, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()