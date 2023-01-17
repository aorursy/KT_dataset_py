# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/data/data/Validation"))

# Any results you write to the current directory are saved as output.
# Any results you write to the current directory are saved as output.

# image classification
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 3, activation = 'softmax'))
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Note: when using the categorical_crossentropy loss, your targets should be in categorical format (e.g. if you have 10 classes, the target for each sample should be a 10-dimensional vector that is all-zeros except for a 1 at the index corresponding to the class of the sample).


# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('../input/data/data/Training',
target_size = (64, 64),
batch_size = 32,
class_mode = 'categorical')


#steps_per_epoch - Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. 
#The default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.
#validation_steps: Only relevant if steps_per_epoch is specified. Total number of steps (batches of samples) to validate before stopping.

import datetime
print("Start time:",datetime.datetime.now())
test_set = test_datagen.flow_from_directory('../input/data/data/Validation',
target_size = (64, 64),
batch_size = 32,
class_mode = 'categorical')

classifier.fit_generator(training_set,
steps_per_epoch = 10,
epochs = 200,
validation_data = test_set,
validation_steps = 10
)
print("End time:",datetime.datetime.now())

from keras.models import load_model

#classifier.save('../input/multiclass_model_latest.h5')  # creates a HDF5 file 'my_model.h5'
#del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
#classifier = load_model('../input/multiclass_model_latest.h5')

# class indices
print(training_set.class_indices)

# test image
import cv2
img = cv2.imread("../input/data/data/Validation/Banana/4_100.jpg")
print("image shape:",img.shape)
img = cv2.resize(img,(64,64))
print("image shape:",img.shape)

best_threshold = [0.4,0.4,0.4]

import numpy as np
img = img.astype('float32')
img = img/255
img = np.expand_dims(img,axis=0)
img.shape
pred = classifier.predict(img)
print("prediction probabilities:",pred)

y_pred = np.array([1 if pred[0,i]>=best_threshold[i] else 0 for i in range(pred.shape[1])])

print("ypred",y_pred)

classes = ['Apple Braeburn','Banana','Dates']
output_class = [classes[i] for i in range(3) if y_pred[i]==1 ]  #extracting actual class name
print("Predicted Class is ",output_class[0])

## 2nd test
import cv2
img = cv2.imread("../input/data/data/Validation/Apple Braeburn/1_100.jpg")
print("image shape:",img.shape)
img = cv2.resize(img,(64,64))
print("image shape:",img.shape)

import numpy as np
img = img.astype('float32')
img = img/255
img = np.expand_dims(img,axis=0)
img.shape
pred = classifier.predict(img)
print("prediction probabilities:",pred)
best_threshold = [0.4,0.4,0.4]
y_pred = np.array([1 if pred[0,i]>=best_threshold[i] else 0 for i in range(pred.shape[1])])

print("ypred",y_pred)

classes = ['Apple Braeburn','Banana','Dates']
output_class = [classes[i] for i in range(3) if y_pred[i]==1 ]  #extracting actual class name
print("Predicted Class is ",output_class)

## 3rd test
import cv2
img = cv2.imread("../input/data/data/Validation/Dates/13_100.jpg")
print("image shape:",img.shape)
img = cv2.resize(img,(64,64))
print("image shape:",img.shape)

import numpy as np
img = img.astype('float32')
img = img/255
img = np.expand_dims(img,axis=0)
img.shape
pred = classifier.predict(img)
print("prediction probabilities:",pred)

y_pred = np.array([1 if pred[0,i]>=best_threshold[i] else 0 for i in range(pred.shape[1])])

print("ypred",y_pred)

classes = ['Apple Braeburn','Banana','Dates']
output_class = [classes[i] for i in range(3) if y_pred[i]==1 ]  #extracting actual class name
print("Predicted Class is ",output_class[0])
