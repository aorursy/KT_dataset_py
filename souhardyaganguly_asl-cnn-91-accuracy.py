# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale = 1./255, 
                             shear_range = 0.2,
                             validation_split = 0.1, 
                             zoom_range = 0.2,  
                             horizontal_flip = True,
                             samplewise_center = True, 
                             samplewise_std_normalization = True)

training_set = datagen.flow_from_directory('/kaggle/input/asl-alphabet-new-24/asl-alphabet/asl_train/asl_alphabet_train/', 
                                           target_size = (64, 64), 
                                           batch_size = 64,
                                           class_mode = 'categorical',
                                           subset = 'training')

test_set = datagen.flow_from_directory('/kaggle/input/asl-alphabet-new-24/asl-alphabet/asl_train/asl_alphabet_train/',
                                       target_size = (64, 64), 
                                       batch_size = 64,
                                       class_mode = 'categorical', 
                                       subset = 'validation')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization
# Initialising the CNN
cnn = Sequential()

#First Layer
cnn.add(Conv2D(filters = 64, kernel_size = (4, 4), 
                      input_shape = (64, 64, 3), activation = 'relu'))

cnn.add(Conv2D(filters = 64, kernel_size = (4, 4), strides = 2,  activation = 'relu'))

cnn.add(Dropout(0.5))

cnn.add(BatchNormalization(axis = 3, momentum = 0.8))

#Second Layer
cnn.add(Conv2D(filters = 128, kernel_size = (4, 4), activation = 'relu'))

cnn.add(Conv2D(filters = 128, kernel_size = (4, 4), strides = 2,  activation = 'relu'))

cnn.add(Dropout(0.5))

cnn.add(BatchNormalization(axis = 3, momentum = 0.8))

#Third Layer

cnn.add(Conv2D(filters = 256, kernel_size = (4, 4), activation = 'relu'))

cnn.add(Conv2D(filters = 256, kernel_size = (4, 4), strides = 2,  activation = 'relu'))

cnn.add(Dropout(0.5))

cnn.add(BatchNormalization(axis = 3, momentum = 0.8))

# Flattening
cnn.add(Flatten())

cnn.add(Dropout(0.5))

# Hidden Layer and Output Layer
cnn.add(Dense(units = 512, activation = 'relu'))
cnn.add(Dense(units = 24, activation = 'softmax'))

#Compiling the CNN
cnn.compile(optimizer= 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
cnn.fit_generator(training_set, steps_per_epoch = 1013, epochs = 15,
                         validation_data = test_set, validation_steps = 113)
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = cnn.evaluate(test_set, batch_size=64)
print("test loss, test acc:", results)
