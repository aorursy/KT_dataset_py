# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import matplotlib.image as mpimg





from keras.utils.np_utils import to_categorical   # for one-hot encoding

from keras.models import Sequential 

from keras.layers import Conv2D,Dense,Flatten,MaxPooling2D,Dropout

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# loading the train and test sets



train_file=pd.read_csv("../input/train.csv")



test_file=pd.read_csv("../input/test.csv")



train_file.head()

# test_file.head()

# separating the training target labels from training inputs



X_train=train_file.drop(columns='label')



Y_train=train_file[['label']]



print(X_train.shape)        # no. of training examples
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])



Y_train=to_categorical(Y_train,num_classes=10)



# normalization of inputs



X_train=X_train/255



test_file=test_file/255



# reshaping into 3D matrices



X_train=X_train.values.reshape(-1,28,28,1)



test_file=test_file.values.reshape(-1,28,28,1)



#splitting the training data into training and validation 



random_seed=2

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state=random_seed)
model=Sequential()



model.add(Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding='same',activation='relu',

                 input_shape=(28,28,1)))

model.add(Conv2D(filters=40,kernel_size=(5,5),strides=(1,1),padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))

model.add(Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())

model.add(Dense(256,activation='relu'))

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))
optimizer=Adam(lr=0.001)



model.compile(optimizer=optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])
reduce_learning_rate=ReduceLROnPlateau(monitor='val_acc',

                                       patience=3,

                                       verbose=1,

                                       factor=0.5,

                                       min_lr=0.00001)



# Saving the model that performed the best on the validation set

checkpoint = ModelCheckpoint(filepath='Model.weights.best.hdf5', save_best_only=True, verbose=1)
datagen=ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images



datagen.fit(X_train)
history=model.fit_generator(datagen.flow(X_train,Y_train,batch_size=128),epochs=20,validation_data=(X_val,Y_val),

                            verbose=1,steps_per_epoch=X_train.shape[0]//128,callbacks=[reduce_learning_rate,checkpoint])
model.load_weights('Model.weights.best.hdf5')



results=model.predict(test_file)



results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)