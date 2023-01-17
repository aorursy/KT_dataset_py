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
# Importing the Keras libraries and packages

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Convolution2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing import image

import numpy as np



# Initialising the CNN

classifier = Sequential()



# Step 1 - Convolution

classifier.add(Convolution2D(32, 3, 3, input_shape = (224, 224, 3), activation = 'relu'))



# Step 2 - Pooling

# classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Adding a second convolutional layer

classifier.add(Convolution2D(32, 3, 3, activation = 'relu')) #220

classifier.add(MaxPooling2D(pool_size = (2, 2))) #110

classifier.add(Convolution2D(32, 3, 1, activation = 'relu')) #108

classifier.add(Convolution2D(32, 1, 3, activation = 'relu')) #106

classifier.add(Convolution2D(32, 3, 1, activation = 'relu')) #104

classifier.add(Convolution2D(32, 1, 3, activation = 'relu')) #102



# Step 3 - Flattening

classifier.add(Flatten())



# Step 4 - Full connection

classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 2, activation = 'softmax'))



# Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



# Part 2 - Fitting the CNN to the images



from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)



# training_set = train_datagen.flow_from_directory('../input/chest_xray/chest_xray/train',

#                                                  target_size = (64, 64),

#                                                  batch_size = 32,

#                                                  class_mode = 'binary')



# test_set = test_datagen.flow_from_directory('dataset/test_set',

#                                             target_size = (64, 64),

#                                             batch_size = 32,

#                                             class_mode = 'binary')



training_set = train_datagen.flow_from_directory('../input/chest_xray/chest_xray/train',

                                                 target_size = (224,224),

                                                 batch_size = 32,

                                                 class_mode = 'categorical')



test_set = test_datagen.flow_from_directory('../input/chest_xray/chest_xray/test',

                                            target_size = (224, 224),

                                            batch_size = 32,

                                            class_mode = 'categorical')



classifier.fit_generator(training_set,

                         steps_per_epoch = 200,

                         epochs = 5,

                         validation_data = test_set,

                         validation_steps = 3000)

classifier.save('model_xray1.h5')

classifier.fit_generator(training_set,

                         steps_per_epoch = 200,

                         epochs = 5,

                         validation_data = test_set,

                         validation_steps = 3000)

classifier.save('model_xray1.h5')
#This cell is to predict the images present in validation set. [[0]] -> NORMAL and [[1]] -> PNUEMONIA

 

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing import image

import numpy as np



# dimensions of our images

img_width, img_height = 224, 224



# load the model we saved

model = load_model('model_xray1.h5')

model.compile(loss='binary_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])



# predicting images

img = image.load_img('../input/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1442-0001.jpeg', target_size=(img_width, img_height))

x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)



images = np.vstack([x])

classes = model.predict_classes(images, batch_size=1)

# print(classes)