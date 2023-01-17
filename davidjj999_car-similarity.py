#An attempt at using a siamese network to measure difference btween cars.  Will eventually develop

#into one-shot learning
#import necessary libraries

import os

from sklearn.metrics import confusion_matrix

import numpy as np

import tensorflow as tf

import tensorflow.keras as keras

import cv2

import matplotlib.pyplot as plt

import math

import random

%matplotlib inline
#set the path of training data

path_root='/kaggle/input/stanford-car-dataset-by-classes-folder/car_data/car_data/train'
#list the classes

class_names=os.listdir(path_root)

class_names.sort()

print(class_names)
#loop to determine biggest/smallest picture in training set and total number of pictures

min_width=9999

min_height=9999

max_width=0

max_height=0

picture_count=0

picture_list=[]

aspect_ratio=[]

for i in class_names:

    path=os.path.join(path_root,i)

    pictures=os.listdir(path)

    pictures.sort()

    for j in pictures:

        img=cv2.imread(os.path.join(path,j),0)

        min_width=min(min_width,np.shape(img)[1])

        min_height=min(min_height,np.shape(img)[0])

        max_width=max(max_width,np.shape(img)[1])

        max_height=max(max_height,np.shape(img)[0])

        picture_count=picture_count+1

        aspect_ratio.append(np.shape(img)[1]/np.shape(img)[0])
plt.hist(aspect_ratio)

plt.show()
#most common aspect ratio.  I used 135x100px for the input image size

np.median(aspect_ratio)
print([min_width,min_height,max_width,max_height,picture_count])
resize_width=135

resize_height=100

print([resize_width,resize_height])
def pair_generator(batch_size,training=True):

    good_pairs=math.ceil(batch_size/2)

    bad_pairs=batch_size-good_pairs

    left_pictures=[]

    right_pictures=[]

    labels=[]

    if training:

        path_root=('/kaggle/input/stanford-car-dataset-by-classes-folder/car_data/car_data/train')

        class_names=os.listdir(path_root)

    else:

        path_root=('/kaggle/input/stanford-car-dataset-by-classes-folder/car_data/car_data/test')

        class_names=os.listdir(path_root)

    while True:

        pair_classes=random.choices(class_names,k=batch_size)

        for i in range(batch_size):

            if i<good_pairs:

                pics=random.sample(os.listdir(os.path.join(path_root,pair_classes[i])),k=2)

                img=cv2.imread(os.path.join(path_root,pair_classes[i],pics[0]))

                img[0]=img[0]/255

                img[1]=img[1]/255

                img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img=cv2.resize(img,(resize_width,resize_height))

                left_pictures.append(img)

                img=cv2.imread(os.path.join(path_root,pair_classes[i],pics[1]))

                img[0]=img[0]/255

                img[1]=img[1]/255

                img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img=cv2.resize(img,(resize_width,resize_height))

                right_pictures.append(img)

                labels.append(1)

            else:

                classes=random.sample(pair_classes,k=2)

                pic1=random.sample(os.listdir(os.path.join(path_root,classes[0])),k=1)

                img=cv2.imread(os.path.join(path_root,classes[0],pic1[0]))

                img[0]=img[0]/255

                img[1]=img[1]/255

                img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img=cv2.resize(img,(resize_width,resize_height))

                left_pictures.append(img)

                pic2=random.sample(os.listdir(os.path.join(path_root,classes[1])),k=1)

                img=cv2.imread(os.path.join(path_root,classes[1],pic2[0]))

                img[0]=img[0]/255

                img[1]=img[1]/255

                img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img=cv2.resize(img,(resize_width,resize_height))

                right_pictures.append(img)

                labels.append(0)

        left_pictures=np.asarray(left_pictures)

        right_pictures=np.asarray(right_pictures)

        labels=np.asarray(labels)

        out=([left_pictures,right_pictures],labels)

        yield out

        left_pictures=[]

        right_pictures=[]

        labels=[]

        
#get 10 pairs to demonstrate the generator

batch_size=10

foo=pair_generator(batch_size)

bar=next(foo)
#verify the shape of the output tensor.  [0][0] is the left image

#expect 10 pairs, 100x135px images, 3 color channels

np.shape(bar[0][0])
#plot the pairs to demonstrate output

count=1

plt.figure(figsize=(40,40))

for i in range(batch_size):

    for j in range(2):

        plt.subplot(batch_size,2,count)

        plt.imshow(bar[0][j][i])

        count=count+1

        plt.xlabel('Truth: '+str(bar[1][i]),fontsize=24)

        plt.xticks([])

        plt.yticks([])

plt.show()
def make_model():

    shape=(100,135,3)

    left_input=keras.Input(shape)

    right_input=keras.Input(shape)

    model=keras.Sequential()

    model.add(keras.layers.Conv2D(32,8,activation='relu',padding='same',input_shape=shape))

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.MaxPooling2D(padding='same'))

    model.add(keras.layers.Conv2D(64,4,activation='relu',padding='same'))

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.MaxPooling2D(padding='same'))

    model.add(keras.layers.Conv2D(128,4,activation='relu',padding='same'))

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.MaxPooling2D(padding='same'))

    model.add(keras.layers.Conv2D(256,4,activation='relu',padding='same'))

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.MaxPooling2D(padding='same'))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(2048,activation='relu'))

    encoded_l=model(left_input)

    encoded_r=model(right_input)

    L1_layer = keras.layers.Lambda(lambda tensors:tf.math.abs(tensors[0] - tensors[1]))

    L1_distance = L1_layer([encoded_l, encoded_r])

    prediction = keras.layers.Dense(1,activation='sigmoid')(L1_distance)

    siamese_net = keras.Model(inputs=[left_input,right_input],outputs=prediction)

    return siamese_net

    
model=make_model()
model.compile(optimizer='adam',loss='BinaryCrossentropy',metrics=['Accuracy'])
model.summary()
prayers=pair_generator(128)
#train the model.  128 batch size times 64 setps is approximately the entire training set.

#100 Epochs is around 4 hours of training.  Note the CPU load from resizing the images

#in the pair generator is by far the limiting factor

model.fit_generator(prayers,epochs=200,steps_per_epoch=64,verbose=1)
#generate a batch to compare the prediction to actual labels visually

batch_size=10

results=pair_generator(batch_size,training=False)

test_data=next(results)

test_x=test_data[0]

test_y=test_data[1]

out=model.predict(test_x,steps=1,verbose=1)
count=1

plt.figure(figsize=(40,40))

for i in range(batch_size):

    for j in range(2):

        plt.subplot(batch_size,2,count)

        plt.imshow(test_x[j][i])

        count=count+1

        plt.ylabel('Truth: '+str(test_y[i]),fontsize=24)

        plt.xlabel('Predicted: '+str(int(out[i][0])),fontsize=24)

        plt.xticks([])

        plt.yticks([])

plt.show()
path_root=('/kaggle/input/stanford-car-dataset-by-classes-folder/car_data/car_data/test')

class_names=os.listdir(path_root)
#find the number of pictures in the test set

picture_count=0

for i in class_names:

    path=os.path.join(path_root,i)

    pictures=os.listdir(path)

    pictures.sort()

    for j in pictures:

        picture_count=picture_count+1

print(picture_count)
#generate arrays of true and predicted values for the entire test set

i=0

final_predicted_y=[]

final_true_y=[]

while i<picture_count:

    results=pair_generator(128,training=False)

    test_data=next(results)

    test_x=test_data[0]

    test_y=test_data[1]

    out=model.predict(test_x,steps=1,verbose=0)

    out=np.reshape(out,128)

    final_predicted_y.append(out)

    final_true_y.append(test_y)

    i=i+128

final_predicted_y=np.asarray(final_predicted_y)

final_predicted_y=np.reshape(final_predicted_y,(-1,1))

final_true_y=np.asarray(final_true_y)

final_true_y=np.reshape(final_true_y,(-1,1))
#confusion matrix of results

foo=confusion_matrix(final_true_y,np.round(final_predicted_y))

print(foo)
robs_class='Ferrari FF Coupe 2012'

robs_car='/kaggle/input/stanford-car-dataset-by-classes-folder/car_data/car_data/train/Ferrari FF Coupe 2012/05423.jpg'
img=cv2.imread(robs_car)

img[0]=img[0]/255

img[1]=img[1]/255

img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img=cv2.resize(img,(resize_width,resize_height))
plt.imshow(img)

plt.show()
def one_shot_generator(batch_size):

    good_pairs=math.ceil(batch_size/2)

    bad_pairs=batch_size-good_pairs

    left_pictures=[]

    right_pictures=[]

    labels=[]

    path_root=('/kaggle/input/stanford-car-dataset-by-classes-folder/car_data/car_data/test')

    class_names=os.listdir(path_root)

    while True:

        for i in range(batch_size):

            if i<good_pairs:

                pics=random.sample(os.listdir(os.path.join(path_root,robs_class)),k=1)

                img=cv2.imread(robs_car)

                img[0]=img[0]/255

                img[1]=img[1]/255

                img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img=cv2.resize(img,(resize_width,resize_height))

                left_pictures.append(img)

                img=cv2.imread(os.path.join(path_root,robs_class,pics[0]))

                img[0]=img[0]/255

                img[1]=img[1]/255

                img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img=cv2.resize(img,(resize_width,resize_height))

                right_pictures.append(img)

                labels.append(1)

            else:

                classes=random.sample(class_names,k=1)

                img=cv2.imread(robs_car)

                img[0]=img[0]/255

                img[1]=img[1]/255

                img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img=cv2.resize(img,(resize_width,resize_height))

                left_pictures.append(img)

                pic2=random.sample(os.listdir(os.path.join(path_root,classes[0])),k=1)

                img=cv2.imread(os.path.join(path_root,classes[0],pic2[0]))

                img[0]=img[0]/255

                img[1]=img[1]/255

                img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img=cv2.resize(img,(resize_width,resize_height))

                right_pictures.append(img)

                labels.append(0)

        left_pictures=np.asarray(left_pictures)

        right_pictures=np.asarray(right_pictures)

        labels=np.asarray(labels)

        out=([left_pictures,right_pictures],labels)

        yield out

        left_pictures=[]

        right_pictures=[]

        labels=[]

        
batch_size=10

one_shot=one_shot_generator(batch_size)

test_data=next(one_shot)

test_x=test_data[0]

test_y=test_data[1]

out=model.predict(test_x,steps=1,verbose=1)
#Lets see how this works

count=1

plt.figure(figsize=(40,40))

for i in range(batch_size):

    for j in range(2):

        plt.subplot(batch_size,2,count)

        plt.imshow(test_x[j][i])

        count=count+1

        plt.ylabel('Truth: '+str(test_y[i]),fontsize=24)

        plt.xlabel('Predicted: '+str(int(out[i][0])),fontsize=24)

        plt.xticks([])

        plt.yticks([])

plt.show()