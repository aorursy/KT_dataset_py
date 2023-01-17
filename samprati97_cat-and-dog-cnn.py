# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# we are not importing libraries used for csv files, as keras deals with all of these

from keras.models import Sequential
from keras.layers import Convolution2D #images are two dimensional. Videos are three dimenstional with time.
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initialize the classifier CNN
classifier = Sequential() #Please note that there is another way to build a mode: Functional API.

#applying convolution operation --> build the convolutional layer
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
#32, 3, 3 --> 32 filters with 3 x 3 for each filter. 
#start with 32 filters, and then create more layers with 64, 128, 256, etc
#expected format of the images.
# 256, 256, 3 --> 3 color channels (RGB), 256 x 256 pixels. But when using CPU, 3, 64, 64 --> due to computational limitation
#Max Pooling --> create a pooling layer
classifier.add(MaxPooling2D(pool_size = (2,2)))
# 2 x 2 size --> commonly used to keep much information.

#Flattening --> creating a long vector.
classifier.add(Flatten()) #no parameters needed.

#classic ANN - full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
#common practice: number of hidden nodes between the number of input nodes and output nodes, and choose powers of 2
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2, 
                                   zoom_range = 0.2, 
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/kaggle/input/cat-and-dog/training_set/', 
                                                    target_size = (64, 64), 
                                                    batch_size = 32,
                                                   class_mode = 'binary')
test_set = test_datagen.flow_from_directory('/kaggle/input/cat-and-dog/test_set/',
                                                target_size = (64, 64),
                                                 batch_size = 32, 
                                                 class_mode = 'binary')

classifier.fit_generator(training_set, 
                         samples_per_epoch = 8005, 
                        nb_epoch = 2, 
                        validation_data = test_set, 
                        nb_val_samples = 2025)

model=Sequential()
model.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Convolution2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Convolution2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(1,activation='sigmoid'))   

model.summary()
model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
model.fit_generator(training_set, 
                         samples_per_epoch = 8005, 
                        nb_epoch = 2, 
                        validation_data = test_set, 
                        nb_val_samples = 2025)
y_pred = model.predict(test_set)
y_pred[1]
type(test_set)
S=64
import os
import cv2
from matplotlib import pyplot as plt

directory = os.listdir("/kaggle/input/cat-and-dog/test_set/test_set/dogs")
print(directory[10])

imgCat = cv2.imread("/kaggle/input/cat-and-dog/test_set/test_set/dogs/" + directory[10])
plt.imshow(imgCat)

imgCat = cv2.resize(imgCat, (S,S))
imgCat = imgCat.reshape(1,S,S,3)

pred = classifier.predict(imgCat)
print("Probability that it is a Cat = ", "%.2f" % (1-pred))


import numpy as np
from keras.preprocessing import image
test_image =image.load_img('/kaggle/input/cat-and-dog/test_set/test_set/cats/cat.4009.jpg',target_size =(64,64))
test_image =image.img_to_array(test_image)
test_image =np.expand_dims(test_image, axis =0)
result = classifier.predict(test_image)
if result[0][0] >= 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)
