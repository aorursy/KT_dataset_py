

#data preprocessing

import pandas as pd

#math operations

import numpy as np

#machine learning



from sklearn.preprocessing import StandardScaler

#hyperparameter optimization

from sklearn.model_selection import GridSearchCV

#support vector machine model

from keras import regularizers

import matplotlib.pyplot as plt

from PIL import Image

import numpy as np

import os

import cv2

import keras

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

            

from random import shuffle

from tqdm import tqdm  

import scipy

import skimage

from skimage.transform import resize

import random
data=[]

labels=[]

Parasitized=os.listdir("../input/cell_images/cell_images/Parasitized/")

for a in Parasitized:

    try:

        image=cv2.imread("../input/cell_images/cell_images/Parasitized/"+a)

        image_from_array = Image.fromarray(image, 'RGB')

        size_image = image_from_array.resize((50, 50))

        data.append(np.array(size_image))

        labels.append(0)

    except AttributeError:

        print("")



Uninfected=os.listdir("../input/cell_images/cell_images/Uninfected/")

for b in Uninfected:

    try:

        image=cv2.imread("../input/cell_images/cell_images/Uninfected/"+b)

        image_from_array = Image.fromarray(image, 'RGB')

        size_image = image_from_array.resize((50, 50))

        data.append(np.array(size_image))

        labels.append(1)

    except AttributeError:

        print("")
Cells=np.array(data)

labels=np.array(labels)
np.save("Cells",Cells)

np.save("labels",labels)
Cells=np.load("Cells.npy")

labels=np.load("labels.npy")
s=np.arange(Cells.shape[0])

np.random.shuffle(s)

Cells=Cells[s]

labels=labels[s]
num_classes=len(np.unique(labels))

len_data=len(Cells)

(x_train,x_test)=Cells[(int)(0.1*len_data):],Cells[:(int)(0.1*len_data)]

x_train = x_train.astype('float32')/255 # As we are working on image data we are normalizing data by divinding 255.

x_test = x_test.astype('float32')/255

train_len=len(x_train)

test_len=len(x_test)
(y_train,y_test)=labels[(int)(0.1*len_data):],labels[:(int)(0.1*len_data)]
#Doing One hot encoding as classifier has multiple classes

y_train=keras.utils.to_categorical(y_train,num_classes)

y_test=keras.utils.to_categorical(y_test,num_classes)
#creating sequential model

model=Sequential()

model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())

#l2 regularizer

model.add(Dense(500,kernel_regularizer=regularizers.l2(0.01),activation="relu"))

model.add(Dense(2,activation="softmax"))#2 represent output layer neurons 

#creating sequential model

model=Sequential()

model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

# 1st dropout

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(500,activation="relu"))

# 2nd dropout

model.add(Dropout(0.2))

model.add(Dense(2,activation="softmax"))#2 represent output layer neurons 

from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

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



datagen.fit(x_train)
from keras.callbacks import EarlyStopping

earlystop= EarlyStopping(monitor='val_acc', patience=3)

epochs = 20 # 

batch_size = 256
#creating sequential model

model=Sequential()

model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

# 1st dropout

model.add(Dropout(0.2))

model.add(Flatten())

#l2 regularizer

model.add(Dense(500,kernel_regularizer=regularizers.l2(0.01),activation="relu"))

# 2nd dropout

model.add(Dropout(0.2))

model.add(Dense(2,activation="softmax"))#2 represent output layer neurons 

model.summary()

# Compiling model

# compile the model with loss as categorical_crossentropy and using adam optimizer you can test result by trying RMSProp as well as Momentum

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model

history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_test,y_test),

                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size

                              , callbacks=[earlystop])
from sklearn.metrics import confusion_matrix

pred = model.predict(x_test)

pred = np.argmax(pred,axis = 1) 

y_true = np.argmax(y_test,axis = 1)
CM = confusion_matrix(y_true, pred)

from mlxtend.plotting import plot_confusion_matrix

fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(5, 5))

plt.show()
Recall=1298/(1298+87)

Recall
Precision = 1298/(1298+48)

Precision