#Importing Libraries

import numpy as np

import tensorflow as tf
#Importing Module for Image Processing

from keras.preprocessing.image import ImageDataGenerator
#Loading and Augmentation of Training Set

train_data = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

train_set = train_data.flow_from_directory('../input/cnn-dataset/CNN_Dataset/training_set', target_size=(64,64),

                                                    batch_size=32, class_mode='binary')
#Loading and Augmentation Test Set

test_data = ImageDataGenerator(rescale=1./255)

test_set = test_data.flow_from_directory('../input/cnn-dataset/CNN_Dataset/test_set', target_size=(64,64), batch_size=32,

                                         class_mode='binary')
#Initialize CNN

ConvNet = tf.keras.models.Sequential()
#Convolution

ConvNet.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))
#Max Pooling

ConvNet.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
#Second Convolution Layer

ConvNet.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))

ConvNet.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
#Flattening

ConvNet.add(tf.keras.layers.Flatten())
#Fully Connected Layer

ConvNet.add(tf.keras.layers.Dense(units=128,activation='relu'))
#Output Layer

ConvNet.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
#Compiling

ConvNet.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#Training

ConvNet.fit(x = train_set, validation_data = test_set, epochs = 30)
#Importing Module for Processing the Image

from keras.preprocessing import image



#Loading the Image(An Image of a Dog) to be Predicted

test_image = image.load_img('../input/cnn-dataset/CNN_Dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))



#Converting to a NumPy array

test_image = image.img_to_array(test_image)



#To input it as a batch just like the Training Set

test_image = np.expand_dims(test_image, axis=0)



#Prediction

result = ConvNet.predict(test_image)
#To check the class indices of Dogs & Cats 

train_set.class_indices
#Since the input of the image is in a batch, 1st index ~ batch index & 2nd index ~ index of the

#ith index in the batch ([0][0] ~ because in this prediction we have 1 batch and 1 image)

if result[0][0]==1:

    Animal = 'Dog'

else:

    Animal = 'Cat'



print(Animal)