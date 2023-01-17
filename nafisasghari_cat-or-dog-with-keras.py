

import numpy as np 

import pandas as pd 



# Input data files are available in the "../input/" directory.



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames[:1]:

        print(os.path.join(dirname, filename))

# Building the CNN



# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# Initialising the CNN

classifier = Sequential()

img_width , img_height = 64, 64



# Step 1 - Convolution

classifier.add(Conv2D(32, (3, 3), input_shape = (img_width , img_height, 3), activation = 'relu'))



# Step 2 - Pooling

classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Adding a second convolutional layer

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))





# Step 3 - Flattening

classifier.add(Flatten())



# Step 4 - Full connection

classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.summary()
# Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the CNN to the images



from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)





training_set = train_datagen.flow_from_directory('/kaggle/input/cat-and-dog/training_set/training_set',

                                                 target_size = (img_width , img_height),

                                                 batch_size = 32,

                                                 class_mode = 'binary')



test_set = test_datagen.flow_from_directory('/kaggle/input/cat-and-dog/test_set/test_set',

                                            target_size = (img_width , img_height),

                                            batch_size = 32,

                                            class_mode = 'binary')



classifier.fit_generator(training_set,

                         steps_per_epoch = 8005, 

                         epochs = 3,

                         validation_data = test_set,

                         validation_steps = 2023) 
