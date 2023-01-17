# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        #print(os.path.join(dirname, filename))

        pass



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from PIL import Image

pil_im = Image.open('../input/logocanal/LOGO PNG.png')

pil_im
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
mypath='/kaggle/input/cat-and-dog/training_set/training_set/cats/'



from os import listdir

from os.path import isfile, join

cats = [mypath+f for f in listdir(mypath) if isfile(join(mypath, f))]

cats_class=['cat' for f in listdir(mypath) if isfile(join(mypath, f))]
mypath='/kaggle/input/cat-and-dog/training_set/training_set/dogs/'



from os import listdir

from os.path import isfile, join

dogs = [mypath+f for f in listdir(mypath) if isfile(join(mypath, f))]

dogs_class=['dogs' for f in listdir(mypath) if isfile(join(mypath, f))]
import pandas as pd

df_cats=pd.DataFrame(cats,columns=['filename'])

df_cats['class']=cats_class



df_dogs=pd.DataFrame(dogs,columns=['filename'])

df_dogs['class']=dogs_class



df=pd.concat([df_dogs,df_cats],axis=0)
df=df.sample(frac=1)
datagen = ImageDataGenerator(width_shift_range=[-100,100],horizontal_flip=True)

it = datagen.flow_from_dataframe(dataframe=df,batch_size=10,image_size=(256, 256))
images,labels=it.next()
print(images[0].shape)
plt.figure(figsize=(100,100))

plt.subplots_adjust(left=0.1, right=0.2, top=0.2, bottom=0.1)



# generate samples and plot

for i in range(9):

    plt.subplot(330 + 1 + i)

    images,labels = it.next()

    image = images[0].astype('uint8')

    plt.imshow(image)

    plt.axis('off')

# show the figure

plt.show()
datagen = ImageDataGenerator(width_shift_range=[-100,100],horizontal_flip=True,rescale=1.0/255.0)

it = datagen.flow_from_dataframe(dataframe=df,batch_size=10,image_size=(256, 256))
import keras,os

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout, Activation

from keras.preprocessing.image import ImageDataGenerator

from keras import layers

import numpy as np
from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D



model = Sequential()

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))



model.add(Dense(2, activation='softmax'))

from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("/kaggle/working/vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, 

                             save_weights_only=False, mode='auto', period=1)

early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')

from keras.optimizers import SGD

opt = SGD(lr=0.001, momentum=0.9)

model.compile(optimizer=opt, loss=keras.losses.binary_crossentropy, metrics=['accuracy'])
history = model.fit_generator(steps_per_epoch=800,generator=it,epochs=40,callbacks=[checkpoint,early])


plt.title('Cross Entropy Loss')

plt.plot(history.history['loss'], color='blue', label='train')



# plot accuracy



plt.title('Classification Accuracy')

plt.plot(history.history['accuracy'], color='blue', label='train')

model.save_weights('/kaggle/working/model.h5')