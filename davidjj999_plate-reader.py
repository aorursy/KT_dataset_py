#A notebook to attempt reading number plates, based on the synthetic turkish license plates dataset on kaggle
#import libraries

import os

import numpy as np

import pandas as pd

import tensorflow.keras as keras

import cv2

import matplotlib.pyplot as plt

import string

import shutil

import tensorflow as tf

import random

%matplotlib inline
#get working directory

cwd=os.getcwd()

print(cwd)
#make a path for training folder

training_path=(cwd+'/training')
#create the training folder

os.mkdir(training_path)
#make a path for test folder

test_path=(cwd+'/test')
#make the test folder

os.mkdir(test_path)
#set the current input path

path='/kaggle/input/synthetic-turkish-license-plates/license-plates/'
#make list of all files in input directory

files=os.listdir(path)
#80/20 split of files for training/test

for f in files:

    if np.random.rand(1)<=0.2:

        shutil.copy(path+f,test_path+'/'+f)

    else:

        shutil.copy(path+f,training_path+'/'+f)

            

        

    
#verify the file copy wored

test_files=os.listdir(test_path)

print(test_files[0:5])
training_files=os.listdir(training_path)

print(training_files[0:5])
#check the length is around 20000

len(test_files)
#check the length is around 80000

len(training_files)
#load an image to practice image manipulation

foo=cv2.imread(training_path+'/'+training_files[0],0)/255
#show the image

plt.imshow(foo,cmap='gray')

plt.show()
#define a scale to downsize, input is way too big

scale=(int(.1*foo.shape[1]),int(.1*foo.shape[0]))

print(scale)
#check the resized image

bar=cv2.resize(foo,scale)

plt.imshow(bar,cmap='gray')

plt.show()
#define a function to one-hot encode the characters

#in the license plate

def string_vectorizer(strng, alphabet=string.printable):

    vector = [[0 if char != letter else 1 for char in alphabet] 

                  for letter in strng]

    return np.asarray(vector)
#check that the encoding worked

bar=string_vectorizer(training_files[0].replace('.png',''))

print(bar)
#define a function to convert one-hot back to string

def get_result(bar):

    out=[]

    for i in range(np.shape(bar)[0]):

        out.append(string.printable[np.argmax(bar[i])])

    out=''.join(out)

    return out
#check that the result function works

foo=get_result(bar)

print(foo)
#define a generator, returns a batch of resized images

#and the corresponding array of encoded characters

def generator(path,batch_size,shuffle):

    files=os.listdir(path)

    count=0

    other_count=0

    x=[]

    y=[]

    while True:

        if shuffle==True:

            random.shuffle(files)

        for file in files:

            img=cv2.imread(path+'/'+file,0)/255

            scale=(int(.1*img.shape[1]),int(.1*img.shape[0]))

            x.append(cv2.resize(img,scale))

            y.append(string_vectorizer(file.replace('.png','') ))

            count=count+1

            if count>=batch_size:

                x=np.asarray(x)

                y=np.asarray(y)

                yield (x,y)

                count=0

                x=[]

                y=[]

                #break

                

        
#test that the generator works

prayers=generator(training_path,1,False)
#get a batch of one from the generator

foo=next(prayers)
#verify output is a tupple

type(foo)
#verify x is a batch of image arrays

type(foo[0])
#verify shape is rescaled image

np.shape(foo[0])
#verify y is a batch of label arrays

type(foo[1])
#verify label shape (9 characters * 100 possible classes per

#character)

np.shape(foo[1])
#define function to make the model

def make_model():

    model=keras.Sequential()

    model.add(keras.layers.Reshape((21,102,1),input_shape=(21,102)))

    model.add(keras.layers.Conv2D(16,3,padding='same',activation='relu',))

    model.add(keras.layers.MaxPooling2D(padding='same'))

    model.add(keras.layers.Conv2D(32,3,padding='same',activation='relu'))

    model.add(keras.layers.MaxPooling2D(padding='same'))

    model.add(keras.layers.Conv2D(32,3,padding='same',activation='relu'))

    model.add(keras.layers.MaxPooling2D(padding='same'))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(900,activation='relu'))

    model.add(keras.layers.Reshape((9,100)))

    model.add(keras.layers.Softmax())

    model.compile(optimizer='adam',loss='CategoricalCrossentropy',metrics=['CategoricalCrossentropy'])

    return model
#make a model

model=make_model()
#view the summary

model.summary()
#generate batch of training data

data=generator(training_path,128,True)
#dont retrain the model every time we save the notebook

model.fit_generator(data,epochs=100,verbose=1,steps_per_epoch=312)
#dont save over old model

model.save(cwd)
#use data generator for batches of test data

test_data=generator(test_path,32,False)

test_labels=generator(test_path,32,False)
#make predicitions for one batch of test data

out=model.predict_generator(test_data,steps=1,verbose=1)
#verify predictions are correct shape

np.shape(out)
#grab the same set of 'true' labels from the generator

labels=next(test_labels)

#data=next(test_data)
#verify shape

np.shape(labels[1])
#verify text 'looks' right

get_result(out[3])
#get the true result for the same plate

get_result(labels[1][3])
#define a figure to compare true images to predicted plate numbers

#performance is not great

plt.figure()

for i in (j+1 for j in range(9)):

    #print(i)

    plt.subplot(3,3,i)

    plt.imshow(labels[0][i],cmap='gray')

    plt.xlabel(get_result(out[i]))

    plt.xticks([])

    plt.yticks([])

plt.show()