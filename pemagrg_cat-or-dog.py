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
trainCats = []
for i in os.listdir("../input/training_set/training_set/cats/"):
    if '(' not in i and '_' not in i:
        i = "../input/training_set/training_set/cats/" + i
        trainCats.append(i)
trainDogs = []
for i in os.listdir("../input/training_set/training_set/dogs/"):
    if '_' not in i and '(' not in i:
        i = "../input/training_set/training_set/dogs/" + i
        trainDogs.append(i)
testCats = []
for i in os.listdir("../input/test_set/test_set/cats/"):
    if '(' not in i and '_' not in i:
        i = "../input/test_set/test_set/cats/" + i
        testCats.append(i)
testDogs = []
for i in os.listdir("../input/test_set/test_set/dogs/"):
    if '(' not in i and '_' not in i:
        i = "../input/test_set/test_set/dogs/" + i
        testDogs.append(i)

Cats, Dogs, All = [], [], []
for i, j in zip(trainCats, trainDogs):
#     Cats.append(i)
#     Dogs.append(j)
    All.append(i)
    All.append(j)
for i, j in zip(testCats, testDogs):
#     Cats.append(i)
#     Dogs.append(j)
    All.append(i)
    All.append(j)

# print("Cats", len(Cats))
# print("Dogs", len(Dogs))
print('All', len(All))
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
#no of filter:32, 3,3 is the shape of each filter
# 64X64 resolution , 3 stands for RGB
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# We take a 2x2 matrix we’ll have minimum pixel loss and get a precise region where the feature are located
classifier.add(Flatten())
"""
Flattening is a very important step to understand. 
What we are basically doing here is taking the 2-D array, 
i.e pooled image pixels and converting them to a one dimensional single vector.
"""
classifier.add(Dense(units = 128, activation = 'relu'))
"""
Dense is the function to add a fully connected layer, ‘units’ is where we define the number of nodes that should be present in this hidden layer, these units value will be always between the number of input nodes and the output nodes but the art of choosing the most optimal number of nodes can be achieved only through experimental tries. Though it’s a common practice to use a power of 2. 
"""
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
                rescale = 1./255,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('../input/training_set/training_set/',
                target_size = (64, 64),
                batch_size = 32,
                class_mode = 'binary')

test_set = test_datagen.flow_from_directory('../input/test_set/test_set',
            target_size = (64, 64),
            batch_size = 32,
            class_mode = 'binary')

classifier.fit_generator(training_set,
steps_per_epoch = 8000,
epochs = 25,
validation_data = test_set,
validation_steps = 2000)
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('../input/training_set/training_set/cats/cat.1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
