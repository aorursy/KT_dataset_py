# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/data/data/"))



import numpy as np

import os

import time

from keras.preprocessing import image

from keras.applications.imagenet_utils import preprocess_input

from keras.layers import Dense, Activation, Flatten

from keras.layers import merge, Input

from keras.models import Model

from keras.utils import np_utils

from sklearn.utils import shuffle

#from sklearn.cross_validation import train_test_split

from keras.utils.vis_utils import plot_model

from keras.applications.vgg16 import VGG16





# Any results you write to the current directory are saved as output.
model = VGG16(include_top=True,weights='imagenet')



print(model.summary())

plot_model(model)







# Loading the training data





# Define data path

data_path = "../input/data/data/"



data_dir_list = os.listdir(data_path)



img_data_list=[]



for dataset in data_dir_list:

	img_list=os.listdir(data_path+'/'+ dataset)

	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))

	for img in img_list:

		img_path = data_path + '/'+ dataset + '/'+ img

		img = image.load_img(img_path, target_size=(224, 224))

		x = image.img_to_array(img)

		#x = np.expand_dims(x, axis=0)

		x = preprocess_input(x)

#		x = x/255

		print('Input image shape:', x.shape)

		img_data_list.append(x)

        

        

img_data = np.array(img_data_list)

#img_data = img_data.astype('float32')

print (img_data.shape)

#img_data=np.rollaxis(img_data,1,0)

#print (img_data.shape)

#img_data=img_data[0]

print (img_data[0].shape)

# Define the number of classes

num_classes = 4

num_of_samples = img_data.shape[0]

labels = np.ones((num_of_samples,),dtype='int64')



labels[0:202]=0

labels[202:404]=1

labels[404:606]=2

labels[606:]=3



from keras.preprocessing.image import ImageDataGenerator

image_gen=ImageDataGenerator(rotation_range=30,width_shift_range=0.1,height_shift_range=0.1,rescale=1/255,

                             shear_range=0.2,zoom_range=0.2

                            ,horizontal_flip=True,fill_mode='nearest')

image_gen.flow_from_directory('../input/data/data/')
# convert class labels to on-hot encoding

Y = np_utils.to_categorical(labels, num_classes)

from sklearn.model_selection import train_test_split

#Shuffle the dataset

x,y = shuffle(img_data,Y, random_state=2)

# Split the dataset

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

#Training the classifier alone

image_input = Input(shape=(224, 224, 3))



model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')

model.summary()

last_layer = model.get_layer('fc2').output

#x= Flatten(name='flatten')(last_layer)

out = Dense(num_classes, activation='softmax', name='output')(last_layer)

custom_vgg_model = Model(image_input, out)

custom_vgg_model.summary()



for layer in custom_vgg_model.layers[:-1]:

	layer.trainable = False



print(custom_vgg_model.layers[2].trainable)



custom_vgg_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

batch_size=16                                                                      

train_gen=image_gen.flow(X_train, y_train, batch_size=32)

test_gen=image_gen.flow(X_test, y_test, batch_size=32)



history=custom_vgg_model.fit_generator(

        train_gen,

        steps_per_epoch=32,

        epochs=12,

        validation_data=test_gen,

        validation_steps=800)

#print(history.history.keys())             

import keras

print(keras.callbacks.History())