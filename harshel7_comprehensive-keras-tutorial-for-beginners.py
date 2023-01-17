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
#read the training data file

df = pd.read_csv('../input/train.csv')

print(df.head())
#let's look at the unique values in the ID column

print(df['Id'].describe())
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize = (15,10))

sns.countplot(y = df['Id'] == 'new_whale', palette = 'Dark2')
#dimension of our original training dataframe

print(df.shape)
#Let's get our x_train and y_train from our dataframe

x_train = df['Image']

y_train = df['Id']
#import all the necessary libraries from the keras API

import keras

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.applications.imagenet_utils import preprocess_input
#define a function to prepare our trianing images

def PrepareTrainImages(dataframe, shape, path):

    

    #obtain the numpy array filled with zeros having the format --> (batch_size, height, width, channels)

    x_train = np.zeros((shape, 100, 100, 3))

    count = 0

    

    for fig in dataframe['Image']:

        

        #load images into images of size 100x100x3

        img = load_img("../input/" + path + "/" + fig, target_size = (100, 100, 3))

        

        #convert images to array

        x = img_to_array(img)

        x = preprocess_input(x)



        x_train[count] = x

        count += 1

    

    return x_train

    
x_train = PrepareTrainImages(df, df.shape[0], 'train')
print(x_train.shape) #we got the data in the format that we need for the CNN model
#let's normalize the data.

x_train[0] # we can see that the pixel values in the following array have large differene in their values

#so it's always better the obtain all the values in the same range

x_train = x_train.astype('float32') / 255 #data normalized
#Let's visualize some of our taining images

plt.figure(figsize = (12,8))

plt.subplot(2, 2, 1)

plt.imshow(x_train[0][:,:,0], cmap = 'gray') #the first image

plt.title(df.iloc[0,0])

plt.xticks([])

plt.yticks([])



plt.subplot(2, 2, 2)

plt.imshow(x_train[100][:,:,0], cmap = 'gray')

plt.title(df.iloc[100,0])

plt.xticks([])

plt.yticks([])



plt.subplot(2, 2, 3)

plt.imshow(x_train[1000][:,:,0], cmap = 'gray')

plt.title(df.iloc[1000,0])

plt.xticks([])

plt.yticks([])



plt.subplot(2, 2, 4)

plt.imshow(x_train[4000][:,:,0], cmap = 'gray')

plt.title(df.iloc[4000,0])

plt.xticks([])

plt.yticks([])
from keras.utils import np_utils #to obtain the one hot encodings of the id values

from sklearn.preprocessing import LabelEncoder #to obtain the unique integer values for each id values
le = LabelEncoder()

y_train = np_utils.to_categorical(le.fit_transform(y_train))
print(y_train[:10])

print(y_train.shape)
#let's start by importing all the necessary libraries for building the CNN model

import keras

from keras.layers import Conv2D

from keras.layers import Activation, BatchNormalization

from keras.layers import MaxPooling2D, Dropout

from keras.layers import Flatten, Dense

from keras.models import Sequential

from keras.optimizers import Adam
#start building the model

model = Sequential()

model.add(Conv2D(32, (3,3), input_shape = (x_train.shape[1:]), padding = 'same'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size =  (2,2)))



model.add(Conv2D(64, (3,3), padding = 'same'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size =  (2,2)))



model.add(Conv2D(128, (3,3), padding = 'same'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size =  (2,2)))



model.add(Flatten())



model.add(Dense(512))

model.add(Activation('relu'))



model.add(Dense(y_train.shape[1]))

model.add(Activation('softmax'))
#looking at the summary for our model

model.summary()
#compile the model

optim = Adam(lr = 0.001) #using the already available learning rate scheduler

model.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])
#fit the model on our dataset

history = model.fit(x_train, y_train, epochs = 30, batch_size = 64)
#let's look how our model performed by plotting the accuracy and loss curves

sns.set(style = 'darkgrid')

plt.figure(figsize = (18, 14))

plt.subplot(2, 1, 1)

plt.plot(range(30), history.history['acc'])

plt.xlabel('EPOCHS')

plt.ylabel('TRAINING ACCURACY')

plt.title('TRAINING ACCURACY vs EPOCHS')



plt.figure(figsize = (18, 14))

plt.subplot(2, 1, 1)

plt.plot(range(30), history.history['loss'])

plt.xlabel('EPOCHS')

plt.ylabel('TRAINING LOSS')

plt.title('TRAINING LOSS vs EPOCHS')
#start building the model

model1 = Sequential()

model1.add(Conv2D(32, (5,5), strides = (2,2), input_shape = (x_train.shape[1:]), padding = 'same'))

model1.add(Activation('relu'))

model1.add(BatchNormalization())

#model1.add(MaxPooling2D(pool_size =  (2,2)))



model1.add(Conv2D(32, (3,3), strides = (2,2), padding = 'same'))

model1.add(Activation('relu'))

model1.add(BatchNormalization())

#model1.add(MaxPooling2D(pool_size =  (2,2)))



# model1.add(Conv2D(128, (3,3), padding = 'same'))

# model1.add(Activation('relu'))

# model1.add(BatchNormalization())

# model1.add(MaxPooling2D(pool_size =  (2,2)))



model1.add(Flatten())



model1.add(Dense(32))

model1.add(Activation('relu'))

model1.add(BatchNormalization())



model1.add(Dense(y_train.shape[1]))

model1.add(Activation('softmax'))



model1.summary()
#compile the model

optim = Adam(lr = 0.001) #using the already available learning rate scheduler

model1.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])
#fit the model on our dataset

history1 = model1.fit(x_train, y_train, epochs = 25, batch_size = 64)
#let's look how our model performed by plotting the accuracy and loss curves

sns.set(style = 'darkgrid')

plt.figure(figsize = (18, 14))

plt.subplot(2, 1, 1)

plt.plot(range(25), history1.history['acc'])

plt.xlabel('EPOCHS')

plt.ylabel('TRAINING ACCURACY')

plt.title('TRAINING ACCURACY vs EPOCHS')



plt.figure(figsize = (18, 14))

plt.subplot(2, 1, 1)

plt.plot(range(25), history1.history['loss'])

plt.xlabel('EPOCHS')

plt.ylabel('TRAINING LOSS')

plt.title('TRAINING LOSS vs EPOCHS')
#start building the model

model2 = Sequential()

model2.add(Conv2D(32, (5,5), input_shape = (x_train.shape[1:]), padding = 'same'))

model2.add(Activation('relu'))

model2.add(BatchNormalization())

model2.add(MaxPooling2D(pool_size =  (2,2), strides = (2,2)))

model2.add(Dropout(0.2))



model2.add(Conv2D(32, (3,3), padding = 'same'))

model2.add(Activation('relu'))

model2.add(BatchNormalization())

model2.add(MaxPooling2D(pool_size =  (2,2), strides = (2,2)))

model2.add(Dropout(0.2))



# model1.add(Conv2D(128, (3,3), padding = 'same'))

# model1.add(Activation('relu'))

# model1.add(BatchNormalization())

# model1.add(MaxPooling2D(pool_size =  (2,2)))



model2.add(Flatten())



model2.add(Dense(128))

model2.add(Activation('relu'))

model2.add(BatchNormalization())

model2.add(Dropout(0.5))



model2.add(Dense(y_train.shape[1]))

model2.add(Activation('softmax'))



model2.summary()
#compile the model

optim = Adam(lr = 0.001) #using the already available learning rate scheduler

model2.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])
#fit the model on our dataset

history2 = model2.fit(x_train, y_train, epochs = 100, batch_size = 64)
#let's look how our model performed by plotting the accuracy and loss curves

sns.set(style = 'darkgrid')

plt.figure(figsize = (18, 14))

plt.subplot(2, 1, 1)

plt.plot(range(100), history2.history['acc'])

plt.xlabel('EPOCHS')

plt.ylabel('TRAINING ACCURACY')

plt.title('TRAINING ACCURACY vs EPOCHS')



plt.figure(figsize = (18, 14))

plt.subplot(2, 1, 1)

plt.plot(range(100), history2.history['loss'])

plt.xlabel('EPOCHS')

plt.ylabel('TRAINING LOSS')

plt.title('TRAINING LOSS vs EPOCHS')
import keras

from keras.applications.mobilenet import MobileNet

from keras.applications.vgg19 import VGG19

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rescale = 1 / 255.,

                            horizontal_flip = True,

                            rotation_range = 10,

                            width_shift_range = 0.1,

                            height_shift_range = 0.1)
mobilenet_model = MobileNet(weights = None, input_shape = (100, 100, 3), classes = 5005)

mobilenet_model.summary()
#compile the model

optim = Adam(lr = 0.001)

mobilenet_model.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])
#fit the model on our data

h2 = mobilenet_model.fit_generator(datagen.flow(x_train, y_train, batch_size = 64), epochs = 300, steps_per_epoch = len(x_train) // 64)
#let's look how our model performed by plotting the accuracy and loss curves

sns.set(style = 'darkgrid')

plt.figure(figsize = (18, 14))

plt.subplot(2, 1, 1)

plt.plot(range(300), h2.history['acc'])

plt.xlabel('EPOCHS')

plt.ylabel('TRAINING ACCURACY')

plt.title('TRAINING ACCURACY vs EPOCHS')



plt.figure(figsize = (18, 14))

plt.subplot(2, 1, 1)

plt.plot(range(300), h2.history['loss'])

plt.xlabel('EPOCHS')

plt.ylabel('TRAINING LOSS')

plt.title('TRAINING LOSS vs EPOCHS')
test_data = os.listdir("../input/test/")

print(len(test_data))
test_data = pd.DataFrame(test_data, columns = ['Image'])

test_data['Id'] = ''
x_test = PrepareTrainImages(test_data, test_data.shape[0], "test")

x_test = x_test.astype('float32') / 255
predictions = mobilenet_model.predict(np.array(x_test), verbose = 1)
for i, pred in enumerate(predictions):

    test_data.loc[i, 'Id'] = ' '.join(le.inverse_transform(pred.argsort()[-5:][::-1]))
test_data.to_csv('model_submission4.csv', index = False)