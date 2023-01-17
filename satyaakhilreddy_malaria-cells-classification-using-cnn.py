# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Keras Imports

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.losses import binary_crossentropy





# Sci-kit learn imports

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle



# Other imports

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import glob

import cv2
# Path array of infected cell images

infected_cells=glob.glob("../input/cell_images/cell_images/Parasitized/*.png")

uninfected_cells=glob.glob("../input/cell_images/cell_images/Uninfected/*.png")
plt.figure(figsize=(15,15))

for i in range(1,6):

    plt.subplot(1,5,i)

    ran=np.random.randint(100)

    plt.imshow(cv2.imread(infected_cells[ran]))

    plt.title('Infected cell')
plt.figure(figsize=(15,15))

for i in range(1,6):

    plt.subplot(1,5,i)

    ran=np.random.randint(100)

    plt.imshow(cv2.imread(uninfected_cells[ran]))

    plt.title('Uninfected cell')
# Create feature and response matrix for the image data

image_arr=list()

label=list()



for i in infected_cells:

    img=cv2.imread(i)

    img_res=cv2.resize(img,(64,64))

    image_arr.append(img_res)

    label.append(1)



for j in uninfected_cells:

    img=cv2.imread(j)

    img_res=cv2.resize(img,(64,64))

    image_arr.append(img_res)

    label.append(0)
# List to Array Conversion and lengths

image_arr=np.array(image_arr)

label=np.array(label)

image_arr.shape, label.shape
# Shuffling of data since all 1s' and 0s' have been appended together



image_arr, label = shuffle(image_arr, label, random_state=0)



# Train-Test split



X_train, X_test, y_train, y_test=train_test_split(image_arr,label,test_size=0.2, random_state=0)
# Image Augementation



## Generic Image Data Generator 

train_generator=ImageDataGenerator(rotation_range=20,width_shift_range=0.25,height_shift_range=0.25,shear_range=0.2,zoom_range=0.3,horizontal_flip=True,vertical_flip=True,rescale=1/255.)

test_generator=ImageDataGenerator(rescale=1/255.)



# Applying generators to training and testing images with additional parameters

train_gen=train_generator.flow(X_train,y_train,batch_size=32,shuffle=False)

test_gen=test_generator.flow(X_test,y_test,batch_size=1,shuffle=False)

# Function to build a neural network



def CNN_neural():

    # Indicates that our model is built using Sequential layers

    model=Sequential()

    

    # First, we add multiples convolution layers to find patterns

    model.add(Conv2D(filters=32,kernel_size=(3,3),strides=1,activation='relu',input_shape=(64,64,3)))

    # Next, we add Pooling layer to reduce the size and find the occurence of feature in the convolution set

    model.add(MaxPooling2D(pool_size=(2,2)))

    # Scales the outputs of previous layers

    model.add(BatchNormalization(axis=-1))

    

    # We repeat the same to create a slightly complex model

    model.add(Conv2D(filters=32,kernel_size=(3,3),strides=1,activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(BatchNormalization(axis=-1))

    

    model.add(Conv2D(filters=32,kernel_size=(3,3),strides=1,activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(BatchNormalization(axis=-1))

    

    # To prevent overfitting

    model.add(Dropout(0.25))

    

    # Now we begin the construction of ANN with the above network output as input

    model.add(Flatten())

    model.add(Dense(256,activation='relu'))

    model.add(BatchNormalization(axis=-1))

    model.add(Dropout(0.25))

    model.add(Dense(1,activation='sigmoid'))

    

    return model

    
model=CNN_neural()

model.compile(loss=binary_crossentropy,

              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),

              metrics=['accuracy'])

model.summary()
# Fitting a model to the data



hist=model.fit_generator(train_gen,steps_per_epoch=len(X_train)/32,epochs=25)
# Diagnostics



N=25

plt.plot(np.arange(0,N),hist.history['loss'],label='Training_loss')

plt.plot(np.arange(0,N),hist.history['acc'],label='Accuracy')

plt.title('Training loss and accuracy')

plt.xlabel('Epochs')

plt.legend(loc='right')
test_err=model.evaluate_generator(test_gen,steps=len(y_test))
# Test Accuracy



print('Loss: ',test_err[0])

print('Accuracy: ',test_err[1])
