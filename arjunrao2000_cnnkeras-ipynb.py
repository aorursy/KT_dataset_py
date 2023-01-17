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
#import the necessary libraries

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import matplotlib.image as mplimg

from matplotlib.pyplot import imshow

import zipfile as zf

import os

import csv

import gc

import operator

import random

from sklearn.model_selection import train_test_split

from numpy import array

from numpy import argmax

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from random import shuffle

from IPython.display import Image

from pathlib import Path



from keras import layers

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.models import Model

from keras.preprocessing import image

from keras.utils import layer_utils

from keras.utils.data_utils import get_file

from keras.applications.imagenet_utils import preprocess_input

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model

from keras.utils import np_utils

import keras.backend as K

from keras.models import Sequential

from keras import optimizers
#reading train csv

df = pd.read_csv('/kaggle/input/train.csv')
df.shape

df.head()
def plot_images_for_filenames(filenames, labels, rows=4):

    imgs = [plt.imread(f'/kaggle/input/train/train/{filename}') for filename in filenames]

    

    return plot_images(imgs, labels, rows)

    

        

def plot_images(imgs, labels, rows=4):

    # Set figure to 13 inches x 8 inches

    figure = plt.figure(figsize=(13, 8))



    cols = len(imgs) // rows + 1



    for i in range(len(imgs)):

        subplot = figure.add_subplot(rows, cols, i + 1)

        subplot.axis('Off')

        if labels:

            subplot.set_title(labels[i], fontsize=16)

        plt.imshow(imgs[i], cmap='gray')
rand_rows = df.sample(frac=1.)[:20]

imgs = list(rand_rows['Image'])

labels = list(rand_rows['Id'])



plot_images_for_filenames(imgs, labels)
import cv2

img = cv2.imread('../input/train/train/00022e1a.jpg')

img.shape
img = plt.imread('../input/train/train/00022e1a.jpg')

plt.imshow(img)
#need to convert y into on-hot encoded labels

def prepareY(Y):



    values = array(Y)

    print(values.shape)

    label_encoder = LabelEncoder()

    integer_encoded = label_encoder.fit_transform(values)

    print(integer_encoded)



    onehot_encoder = OneHotEncoder(sparse=False)

    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    print(onehot_encoded)



    y = onehot_encoded

    print(y.shape)

    return y, label_encoder
Y = df['Id']



print("Shape Y-train: ", Y.shape)



#The next lines are used for testing - not necessary for the code

#labels = trainData['Id'].unique()

#print("Labels in data set: ", labels.shape)

#labelsTrain = Y.unique()

#print("Labels in training set: ", labelsTrain.shape)



y, label_encoder = prepareY(Y)
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_dataframe(

df,

directory = '../input/train/train',

x_col = 'Image',

y_col = 'Id',

target_size = (100,100),

class_mode = 'categorical'

)
mod = Sequential()



mod.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', input_shape = (100, 100, 3)))



mod.add(BatchNormalization(axis = 3, name = 'bn0'))

mod.add(Activation('relu'))



mod.add(MaxPooling2D((2, 2), name='max_pool'))

mod.add(Conv2D(64, (3, 3), strides = (1,1), name="conv1"))

mod.add(Activation('relu'))

mod.add(AveragePooling2D((3, 3), name='avg_pool'))



mod.add(Flatten())

mod.add(Dense(500, activation="relu", name='rl'))

mod.add(Dropout(0.8))

mod.add(Dense(4251, activation='softmax', name='sm'))



print(mod.output_shape)



#opt = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

mod.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
mod.summary()
train_generator
history = mod.fit_generator(

train_generator,

steps_per_epoch = 2000,

epochs = 15,

validation_steps = 64

)
mod.save('CNNkeras.h5')
plt.plot(history.history['acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.show()
def prepareImages(data, m, dataset):

    

    print("Preparing images")

    

    X_train = np.zeros((m, 100, 100, 3))

    

    count = 0

    

    for fig in data['Image']:

        #load images into images of size 100x100x3

        img = image.load_img("../input/"+dataset+"/"+fig, target_size=(100, 100, 3))

        x = image.img_to_array(img)

        #x = np.expand_dims(x, axis=0)

        x = preprocess_input(x)

        X_train[count] = x

        if (count%500 == 0):

            print("Processing image: ", count+1, ", ", fig)

        count += 1

    

    count = 0

    

    print("Finished!")

            

    return X_train
col = ['Image']

test_dir = os.listdir('/kaggle/input/test/test')

test = pd.DataFrame(test_dir,columns = col)

test.head()
print(len(test_dir))
test = os.listdir("../input/test/test")

print(len(test))



#separate data into different DataFrames due to memory constraints

col = ['Image']

testData1 = pd.DataFrame(test[0:3899], columns=col)

testData2 = pd.DataFrame(test[3900:7799], columns=col)

testData3 = pd.DataFrame(test[7800:11699], columns=col)

testData4 = pd.DataFrame(test[11700:15609], columns=col)

testData = pd.DataFrame(test, columns=col)
gc.collect()

X = prepareImages(testData1, 3900, "test/test")

X /= 255
predictions1 = mod.predict(np.array(X), verbose=1)

gc.collect()
X = prepareImages(testData2, 3900, "test/test")

X /= 255

predictions2 = mod.predict(np.array(X), verbose=1)

gc.collect()
X = prepareImages(testData3, 3900, "test/test")

X /= 255

predictions3 = mod.predict(np.array(X), verbose=1)

gc.collect()
X = prepareImages(testData4, 3910, "test/test")

X /= 255

predictions4 = mod.predict(np.array(X), verbose=1)

gc.collect()
predictions = np.concatenate((predictions1, predictions2), axis=0)

predictions = np.concatenate((predictions, predictions3), axis=0)

predictions = np.concatenate((predictions, predictions4), axis=0)

gc.collect()

print(predictions.shape)

print(predictions.shape)



copy_pred = np.copy(predictions)

idx = np.argmax(copy_pred, axis=1)

copy_pred[:,idx] = 0

idx2 = np.argmax(copy_pred, axis=1)

copy_pred[:, idx2] = 0

idx3 = np.argmax(copy_pred, axis=1)

copy_pred[:, idx3] = 0

idx4 = np.argmax(copy_pred, axis=1)

copy_pred[:, idx4] = 0

idx5 = np.argmax(copy_pred, axis=1)
#convert the one-hot vectors to their names

results = []



print(idx[0:10])

print(idx2[0:10])

print(idx3[0:10])

print(idx4[0:10])

print(idx5[0:10])

threshold = 0.05 #threshold - only consider answers with a probability higher than it

for i in range(0, predictions.shape[0]):

#for i in range(0, 10):

    each = np.zeros((4251, 1))

    each2 = np.zeros((4251, 1))

    each3 = np.zeros((4251, 1))

    each4 = np.zeros((4251, 1))

    each5 = np.zeros((4251, 1))

    if((predictions[i, idx5[i]] > threshold)):

        each5[idx5[i]] = 1

        each4[idx4[i]] = 1

        each3[idx3[i]] = 1

        each2[idx2[i]] = 1

        each[idx[i]] = 1

        tags = [label_encoder.inverse_transform([argmax(each)])[0], label_encoder.inverse_transform([argmax(each2)])[0], label_encoder.inverse_transform([argmax(each3)])[0], label_encoder.inverse_transform([argmax(each4)])[0], label_encoder.inverse_transform([argmax(each5)])[0]]

    else:

        if((predictions[i, idx4[i]] > threshold)):

            print(predictions[i, idx4[i]])

            each4[idx4[i]] = 1

            each3[idx3[i]] = 1

            each2[idx2[i]] = 1

            each[idx[i]] = 1

            tags = [label_encoder.inverse_transform([argmax(each)])[0], label_encoder.inverse_transform([argmax(each2)])[0], label_encoder.inverse_transform([argmax(each3)])[0], label_encoder.inverse_transform([argmax(each4)])[0]]

        else:

            if((predictions[i, idx3[i]] > threshold)):

                each3[idx3[i]] = 1

                each2[idx2[i]] = 1

                each[idx[i]] = 1

                tags = [label_encoder.inverse_transform([argmax(each)])[0], label_encoder.inverse_transform([argmax(each2)])[0], label_encoder.inverse_transform([argmax(each3)])[0]]

            else:

                if((predictions[i, idx2[i]] > threshold)):

                    each2[idx2[i]] = 1

                    each[idx[i]] = 1

                    tags = [label_encoder.inverse_transform([argmax(each)])[0], label_encoder.inverse_transform([argmax(each2)])[0]]

                else:

                    each[idx[i]] = 1

                    tags = label_encoder.inverse_transform([argmax(each)])[0]

    results.append(tags)
#write the predictions in a file to be submitted in the competition.

myfile = open('output.csv','w')



column= ['Image', 'Id']



wrtr = csv.writer(myfile, delimiter=',')

wrtr.writerow(column)



for i in range(0, testData.shape[0]):

    pred = ""

    if(len(results[i])==5):

        if (results[i][4]!=results[i][0]):

            pred = results[i][0] + " " + results[i][1] + " " + results[i][2] + " " + results[i][3] + " " + results[i][4]

        else:

            pred = results[i][0] + " " + results[i][1] + " " + results[i][2] + " " + results[i][3]

    else:

        if(len(results[i])==4):

            pred = results[i][0] + " " + results[i][1] + " " + results[i][2] + " " + results[i][3]

        else:

            if(len(results[i])==3):

                pred = results[i][0] + " " + results[i][1] + " " + results[i][2]

            else:

                if(len(results[i])==2):

                    pred = results[i][0] + " " + results[i][1]

                else:

                    pred = results[i]

            

    result = [testData['Image'][i], pred]

    #print(result)

    wrtr.writerow(result)

    

myfile.close()