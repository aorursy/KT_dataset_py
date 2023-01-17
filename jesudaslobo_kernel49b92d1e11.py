# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/dataset/dataset"))



# Any results you write to the current directory are saved as output.


# Part 1 - Building the CNN



# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense
# Initialising the CNN

classifier = Sequential()
# Step 1 - Convolution

classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
# Step 2 - Pooling

classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer

classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening

classifier.add(Flatten())
# Step 4 - Full connection

classifier.add(Dense(output_dim = 128, activation = 'relu'))

classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
# Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Part 2 - Fitting the CNN to the images



from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory('../input/dataset/dataset/training_set',

                                                 target_size = (64, 64),

                                                 batch_size = 32,

                                                 class_mode = 'binary')



test_set = test_datagen.flow_from_directory('../input/dataset/dataset/test_set',

                                            target_size = (64, 64),

                                            batch_size = 32,

                                            class_mode = 'binary')
classifier.fit_generator(training_set,

                         samples_per_epoch = 8000,

                         nb_epoch = 5,

                         validation_data = test_set,

                         nb_val_samples = 2000)
from keras.preprocessing import image

test_image = image.load_img('../input/dataset/dataset/test_set/dogs/dog.4603.jpg', target_size = (64, 64))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

result = classifier.predict(test_image)

training_set.class_indices
if result[0][0] == 1:

    prediction = 'dog'

else:

    prediction = 'cat'

print(prediction)