

import pandas as pd

import numpy as np

import os

import keras

import matplotlib.pyplot as plt

from keras.layers import Dense,GlobalAveragePooling2D

from keras.applications import MobileNet

from keras.preprocessing import image

from keras.applications.mobilenet import preprocess_input

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model

from keras.optimizers import Adam





from keras.preprocessing.image import img_to_array

import random

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Convolution2D,BatchNormalization

from keras.layers import MaxPooling2D,Dropout

from keras.layers import Flatten

from keras.layers import Dense

import cv2

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split



import warnings

warnings.filterwarnings("ignore")



import math

from keras.preprocessing.image import ImageDataGenerator





base_model=MobileNet(weights='imagenet',include_top=False, input_shape=(128, 128, 3)) #imports the mobilenet model and discards the last 1000 neuron layer.



x=base_model.output

x=GlobalAveragePooling2D()(x)

x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.

# x=Dense(1024,activation='relu')(x) #dense layer 2

x=Dense(512,activation='relu')(x) #dense layer 3

preds=Dense(8,activation='softmax')(x) #final layer with softmax activation



model=Model(inputs=base_model.input,outputs=preds)

model.summary()



for layer in model.layers[:-8]:

    layer.trainable=False

for layer in model.layers[-8:]:

    layer.trainable=True

    


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

print(os.listdir("../input/"))
data=[]

labels=[]

random.seed(42)

imagePaths = sorted(list(os.listdir("../input/data/natural_images/")))

random.shuffle(imagePaths)

print(imagePaths)
for img in imagePaths:

    path=sorted(list(os.listdir("../input/data/natural_images/"+img)))

    for i in path:

        image = cv2.imread("../input/data/natural_images/"+img+'/'+i)

        image = cv2.resize(image, (128,128))

        image = img_to_array(image)

        data.append(image)



        l = label = img

        labels.append(l)


data = np.array(data, dtype="float32") / 255.0

labels = np.array(labels)

mlb = LabelBinarizer()

labels = mlb.fit_transform(labels)

print(labels[0])



(x_train,x_test,y_train,y_test)=train_test_split(data,labels,test_size=0.4,random_state=42)

print(x_train.shape, x_test.shape)



model.fit(x_train, y_train, epochs=2)



y_pred = model.predict(x_test)



total = 0

accurate = 0

accurate_index = []

wrong_index = []



for i in range(len(y_pred)):

    if np.argmax(y_pred[i]) == np.argmax(y_test[i]):

        accurate += 1

        accurate_index.append(i)

    else:

        wrong_index.append(i)

        

    total += 1

    

    

print('Total test data:', total, '\taccurately predicted data:', accurate, '\t wrongly predicted data: ', total - accurate)

print('Accuracy:', round(accurate/total*100, 3), '%')



labelss=['airplane','car','cat','dog','flower','fruit','motorbike','person']

# labelss=['car','cat','dog','person']

im_idx = random.sample(wrong_index, k=9)



nrows = 3

ncols = 3

fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(15, 12))



n = 0

for row in range(nrows):

    for col in range(ncols):

            ax[row,col].imshow(x_test[im_idx[n]])

            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(labelss[np.argmax(y_pred[im_idx[n]])], labelss[np.argmax(y_test[im_idx[n]])]))

            n += 1



plt.show()



train_datagen = ImageDataGenerator(zoom_range = 0.2, horizontal_flip=True, shear_range=0.2)



batch_size = 50

model.fit_generator(train_datagen.flow(x_train,y_train,batch_size=batch_size),

                    steps_per_epoch=math.ceil(len(x_train)//batch_size), epochs = 2)

model.save('CNN_imagegenerator-natural-images.h5')


y_pred = model.predict(x_test)



total = 0

accurate = 0

accurate_index = []

wrong_index = []



for i in range(len(y_pred)):

    if np.argmax(y_pred[i]) == np.argmax(y_test[i]):

        accurate += 1

        accurate_index.append(i)

    else:

        wrong_index.append(i)

        

    total += 1

    

    

print('Total test data;', total, '\taccurately predicted data:', accurate, '\t wrongly predicted data: ', total - accurate)

print('Accuracy:', round(accurate/total*100, 3), '%')

labelss=['airplane','car','cat','dog','flower','fruit','motorbike','person']

# labelss=['car','cat','dog','person']

im_idx = random.sample(wrong_index, k=9)



nrows = 3

ncols = 3

fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(15, 12))



n = 0

for row in range(nrows):

    for col in range(ncols):

            ax[row,col].imshow(x_test[im_idx[n]])

            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(labelss[np.argmax(y_pred[im_idx[n]])], labelss[np.argmax(y_test[im_idx[n]])]))

            n += 1



plt.show()
labelss=['airplane','car','cat','dog','flower','fruit','motorbike','person']

# labelss=['car','cat','dog','person']

im_idx = random.sample(accurate_index, k=9)



nrows = 3

ncols = 3

fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(15, 12))



n = 0

for row in range(nrows):

    for col in range(ncols):

            ax[row,col].imshow(x_test[im_idx[n]])

            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(labelss[np.argmax(y_pred[im_idx[n]])], labelss[np.argmax(y_test[im_idx[n]])]))

            n += 1



plt.show()