# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#ok

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory. 

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        y=0

        #print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import cv2

import matplotlib.image as image

import matplotlib.pyplot as plt

%matplotlib inline



import os

import random

import gc
train_dir='/kaggle/input/dogs-vs-cats-redux-kernels-edition/train'



test_dir='/kaggle/input/dogs-vs-cats-redux-kernels-edition/test'

train_dogs=[train_dir+'/{}'.format(i) for i in os.listdir(train_dir) if 'dog' in i]

train_cats=[train_dir+'/{}'.format(i) for i in os.listdir(train_dir) if 'cat' in i]
test_imgs=[train_dir+'/{}'.format(i) for i in os.listdir(train_dir) ]

train_imgs=train_dogs[:2000]+train_cats[:2000]



random.shuffle(train_imgs)

"""

for img in train_imgs[0:10]:

    myimage=image.imread(img)

    imgplot=plt.imshow(myimage)

    plt.show()

   



""" 

    

def read_and_processCD(list_of_images):

    X=[]

    y=[]

    for image in list_of_images:

        X.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR),(150,150),interpolation=cv2.INTER_CUBIC))

        if 'dog.' in image:

            y.append(1)

        elif 'cat.' in image:

            y.append(0)

    return X,y

X,y=read_and_processCD(train_imgs)        

print(train_imgs[0])
for img in X[0:4]:

    #myimage=image.imread(img)

    imgplot=plt.imshow(img)

    plt.show()

    
print(y[0],y[1],y[2],y[3])
import seaborn as sns

X=np.array(X)

y=np.array(y)

sns.countplot(y)





print(X.shape)

print(y.shape)

from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=2)
batch_size=32

from keras import layers

from keras import models

from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import img_to_array, load_img









model=models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512,activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512,activation='relu'))

model.add(layers.Dense(1,activation='sigmoid'))

model.summary()



model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
train_datagen = ImageDataGenerator(rescale=1./255,   #Scale the image between 0 and 1

                                    rotation_range=40,

                                    width_shift_range=0.2,

                                    height_shift_range=0.2,

                                    shear_range=0.2,

                                    zoom_range=0.2,

                                    horizontal_flip=True,)



val_datagen = ImageDataGenerator(rescale=1./255) 
train_generator=train_datagen.flow(X_train,y_train,batch_size=batch_size)

val_generator=val_datagen.flow(X_val,y_val,batch_size=batch_size)

ntrain = len(X_train)

nval = len(X_val)
history=model.fit_generator(train_generator,

                           steps_per_epoch=ntrain//batch_size,

                           epochs=1,

                           validation_data=val_generator,

                           validation_steps=nval//batch_size)

                           

#Save the model

model.save_weights('model_wieghts.h5')

model.save('model_keras.h5')

#model.save_weights('model_wieghts_test.h5')

#model.save('model_keras_test.h5')
model2= models.load_model('model_keras_test.h5')
model2= models.load_model('model_keras_test.h5')
import os

for dirname, _, filenames in os.walk('../working'):

    print('checking...')

    for filename in filenames:

        print(os.path.join(dirname, filename))
os.walk('../')