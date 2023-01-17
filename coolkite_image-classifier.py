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
#important steps in image classification

'''

1. convulation - uses a filter to find certain features like vertical and horizontal edges

2. pooling - reduces size of the image. reduces number of required parameters

3. flattening - converts the whole data into a one dimensional matrix to make a vector

4. Full connection - every neuron in one layer is connected to every other neuron in the next layer

'''
# important modules 

# we could have just imported keras to make it work but I wanted to specifically show which methods are really NEEDED

from keras.models import Sequential   #One of the two ways to initialize a neural network model. The other way: graphs



from keras.layers import Conv2D       #Used to perform the convolution operation



from keras.layers import MaxPooling2D #Used to perform the pooling



from keras.layers import Flatten      #Used to flatten out the data into a vector



from keras.layers import Dense        #Used to fully connect a neuron
# Make an object of Sequential

classify = Sequential()
# convolution

filters = 32             #Used to suppress the non-required parts of a picture

filtershape = (3,3)      #Size of filter



classify.add(Conv2D(filters, filtershape, input_shape = (64, 64, 3), activation = 'relu'))
# pooling

# We take a 2*2 matrix from the 3*3 matrix to minimize the number of pixels lost

classify.add(MaxPooling2D(pool_size = (2, 2)))
#flatten

classify.add(Flatten())
#Full connection. Common practice to use a power of 2. Get best result by trial and error. 

classify.add(Dense(units = 128, activation = 'relu'))
#initalize the output layer



classify.add(Dense(units = 1, activation = 'sigmoid'))
#time to compile

#optimizer - Adams is just a type of optimizer. Like gradient descent

#loss - binary_crossentropy. Need to classify either as a dog or cat (binary)

#metrics - based on the accuracy of the pictures

classify.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# before we use the pictures we will alter them a bit. Blur them, color invert them so that program can identift different vaiations of the picture.

from keras.preprocessing.image import ImageDataGenerator





train_datagen = ImageDataGenerator(rescale = 1./255, #rescales the image by a factor

shear_range = 0.2,                                   #increase intensity

zoom_range = 0.2,                                    #how far to zoom increase the image

horizontal_flip = True)                              #flip the picture over the x-axis





test_datagen = ImageDataGenerator(rescale = 1./255)  #rescaling

training_set = train_datagen.flow_from_directory('/kaggle/input/cat-and-dog/training_set/training_set/', target_size = (64, 64),

batch_size = 32,

class_mode = 'binary')





test_set = test_datagen.flow_from_directory('/kaggle/input/cat-and-dog/test_set/test_set/',

target_size = (64, 64),

batch_size = 32,

class_mode = 'binary')
#fit the data

classify.fit_generator(training_set,

steps_per_epoch = 8005,     #number of images

epochs = 5,                #number of time iteration takes place

validation_data = test_set, #the images it is going to compare its results to

validation_steps = 2023)    #number of images in test set
#make predictions

import numpy as np

from keras.preprocessing import image

test_image = image.load_img('/kaggle/input/dog-or-cat/dog1.jpg', target_size = (64, 64))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

result = classify.predict(test_image)

training_set.class_indices

if result[0][0] == 1:

    prediction = 'dog'

else:

    prediction = 'cat'

print(prediction)