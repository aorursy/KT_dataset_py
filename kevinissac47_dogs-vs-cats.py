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
import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import os

import random

import gc
train_dir = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/train'

test_dir = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test'



train_dogs=['/kaggle/input/dogs-vs-cats-redux-kernels-edition/train/{}'.format(i) for i in os.listdir(train_dir) if 'dog' in i]

train_cats=['/kaggle/input/dogs-vs-cats-redux-kernels-edition/train/{}'.format(i) for i in os.listdir(train_dir) if 'cat' in i]



test_imgs = ['/kaggle/input/dogs-vs-cats-redux-kernels-edition/test/{}'.format(i) for i in os.listdir(test_dir)]

train_imgs = train_dogs[:2000] + train_cats[:2000]



random.shuffle(train_imgs)



del train_dogs

del train_cats

gc.collect()

train_imgs
import matplotlib.image as mpimg

for ima in train_imgs[0:3]:

    img = mpimg.imread(ima)

    imgplot = plt.imshow(img)

    plt.show()
#declare image dimmension to make all images the same size

nrows = 150

ncolumns = 150

channels = 3 # for 3 color image 1 for grayscale



def read_and_process_image(list_of_images):

    """

    Returns two arrays

    X is  array of resized images

    y is array of labels

    """

    X =[] #images

    y=[] #labels

    

    for image in list_of_images:

        X.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR), (nrows,ncolumns), interpolation=cv2.INTER_CUBIC))# read the images get the laabels

        if 'dog.' in image:

            y.append(1)

        elif 'cat.' in image:

            y.append(0)

    return X,y

                
X,y = read_and_process_image(train_imgs)
plt.figure(figsize=(20,10))

columns = 5

for i in range(columns):

    plt.subplot(5/columns + 1,columns,i+1)

    plt.imshow(X[i])
import seaborn as sns

del train_imgs

gc.collect()



#converrt list into numpy arrays

X = np.array(X)

y = np.array(y)



sns.countplot(y)

plt.title('Labels for cats and dogs')
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.20,random_state=2)

del X

del y

gc.collect()



ntrain = len(X_train)

ntest = len(X_test)



#batch_size should be always be a factor of 2,4,8,16...

batch_size = 32

from keras.applications import InceptionResNetV2

conv_base = InceptionResNetV2(weights='imagenet',include_top=False,input_shape=(150,150,3))
conv_base.summary()
from keras import layers,models,optimizers

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import img_to_array,load_img
model = models.Sequential()

model.add(conv_base)

model.add(layers.Flatten())

model.add(layers.Dense(256,activation='relu'))

model.add(layers.Dense(1,activation='sigmoid'))

model.summary()
print(len(model.trainable_weights))

conv_base.trainable=False

print(len(model.trainable_weights))

#model = models.Sequential()

#model.add(layers.Conv2D(32,(3,3),activation = 'relu',input_shape=(150,150,3)))

#model.add(layers.MaxPooling2D((2,2)))

#model.add(layers.Conv2D(64,(3,3),activation='relu'))

#model.add(layers.MaxPooling2D((2,2)))

#model.add(layers.Conv2D(128,(3,3),activation='relu'))

#model.add(layers.MaxPooling2D((2,2)))

#model.add(layers.Conv2D(128,(3,3),activation='relu'))

#model.add(layers.MaxPooling2D((2,2)))

#model.add(layers.Flatten())

#model.add(layers.Dropout(0.5))

#model.add(layers.Dense(512,activation='relu'))

#model.add(layers.Dense(1,activation='sigmoid'))

#model.summary()
#use RMSprop with learning rate 0.0001

#use binary cross entropy because its a binary classification

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=2e-5),metrics=['acc'])
train_datagen = ImageDataGenerator(rescale=1./255, #scale the image between 0 and 1

                                   rotation_range=40,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   horizontal_flip=True,)

val_datagen=ImageDataGenerator(rescale=1./255)
#create image generators

train_generator = train_datagen.flow(X_train,y_train,batch_size=batch_size)

val_generator = val_datagen.flow(X_test,y_test,batch_size=batch_size)

#train 64 epochs with 100 steps each epoch

history = model.fit_generator(train_generator,

                             steps_per_epoch=ntrain//batch_size,

                             epochs=20,

                             validation_data=val_generator,

                             validation_steps=ntest//batch_size)

model.save_weights('model_weights.h5')

model.save('model_keras.h5')
#plot train and test curve

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1,len(acc) + 1)



#train and validation accuracy



plt.plot(epochs, acc, 'b', label='Training accuracy')

plt.plot(epochs, val_acc, 'r', label='Validation accuracy')

plt.title('Training and Validation accuracy')

plt.legend()

plt.figure()



#Train and validatoin loss

plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs,val_loss, 'r', label='Validation Loss')

plt.title('Training and validation Accuracy')

plt.legend()

plt.show()
X_test,y_test = read_and_process_image(test_imgs[0:10])

x = np.array(X_test)

test_datagen = ImageDataGenerator(rescale=1./255)
i=0

test_labels=[]

plt.figure(figsize=(30,20))

for batch in test_datagen.flow(x,batch_size=1):

    pred = model.predict(batch)

    if pred>0.5:

        test_labels.append('dog')

    else:

        test_labels.append('cat')

    plt.subplot(5/columns + 1,columns,i + 1)

    plt.title('This is a ' + test_labels[i])

    imgplot = plt.imshow(batch[0])

    i += 1

    if i%10==0:

        break

    plt.show
