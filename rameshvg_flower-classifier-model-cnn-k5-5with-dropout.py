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
 # Building the CNN

 

import keras

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import Dropout

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense



# Initialising the CNN

my_cnn=Sequential()



## Step 1 - Convolution

my_cnn.add(Conv2D(32,(5,5),input_shape=(32,32,3), activation='relu'))



## Step 1 - Pooling

my_cnn.add(MaxPooling2D(pool_size=(2,2)))



## Adding a second Convolution layer

my_cnn.add(Conv2D(16,(3,3), activation='relu'))

my_cnn.add(MaxPooling2D(pool_size=(2,2)))



# Flattening

my_cnn.add(Flatten())



# Full connection

my_cnn.add(Dense(units=1024,activation='relu'))

my_cnn.add(Dense(units=512,activation='relu'))

my_cnn.add(Dense(units=128,activation='relu'))



# adding a dropout layer

my_cnn.add(Dropout(0.5))



## Adding a loss layer

my_cnn.add(Dense(units=17,activation='softmax'))

#17 output classes exist 



# Compiling the CNN

#from keras.optimizers import RMSprop, SGD



#my_cnn.compile(loss = 'categorical_crossentropy',optimizer = RMSprop(lr = 0.001), metrics = ['accuracy'])

my_cnn.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])



# Fitting the CNN to the images



from keras.preprocessing.image import ImageDataGenerator



train_model=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)



test_model=ImageDataGenerator(rescale=1./255)



train_set=train_model.flow_from_directory('../input/flowers/Flowers/Training',target_size=(32,32),batch_size=16,

                                          class_mode='categorical',shuffle=True)

test_set=test_model.flow_from_directory('../input/flowers/Flowers/Test',target_size=(32,32),batch_size=16,

                                        class_mode='categorical',shuffle=False)



train_set.class_indices

my_cnn.fit_generator(train_set,steps_per_epoch=1071/16,epochs=10,

                     validation_data=test_set,validation_steps=272/16)



my_cnn.summary()



from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



training_set = train_datagen.flow_from_directory('../input/flowers/Flowers/Training',

                                                 target_size = (32,32),

                                                 batch_size = 16,

                                                 class_mode = 'categorical')



import numpy as np

from keras.preprocessing import image

test_image = image.load_img('../input/flowers/Flowers/Prediction/Pansy.jpg',

                            target_size = (32, 32))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

result = my_cnn.predict_classes(test_image)

print(result)

training_set.class_indices

print(training_set.class_indices)



inID = result[0]  

class_dictionary =training_set.class_indices  

inv_map = {v: k for k, v in class_dictionary.items()}  

label = inv_map[inID]  

   

 # get the prediction label  

print(" prediction Image ID: {}, Label: {}".format(inID, label))  




