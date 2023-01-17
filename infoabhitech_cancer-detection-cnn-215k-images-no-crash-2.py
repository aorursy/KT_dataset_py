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
import keras

from keras.preprocessing import image

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D



from skimage.io import imread

from skimage.io import imshow



import os
train = pd.read_csv("../input/train_labels.csv")
train.head()
print("Number of training smaples -->" ,len(train))
# Function to generate full path of image file



def train_func_image_file(x):

    folder = '../input/train/'

    path = folder + x + '.tif'

    return path
# Create image path column in frame



train['path'] = train['id'].apply(train_func_image_file)
print(train['path'][0])
# Read image file using skimage imread functionality

# Loading all training samples might blow off kernel due to limited memory , so taking maximum possible data



train['image'] = train['path'][0:215000].map(imread)
print(imshow(train['image'][1]))
# Function to crop image , to reduce memory usage but maintaining target area of image 30x30



def crop(x):

    return x[24:72, 24:72]
# Create new column for image crop



train['image_crop'] = train['image'][0:215000].map(crop)
print("Cropped image" ,imshow(train['image_crop'][1]))
print("Dimension of image --->" ,train['image'][0].shape)
print("Dimension of crop image --->" ,train['image_crop'][0].shape)
# Drop unwanted columns to release space

train = train.drop(['path'], axis=1)
train = train.drop(['image'], axis=1)
# Garbage collector to release memory



import gc; 

gc.collect()
# Create training array for individual image



x_train = np.stack(list(train.image_crop.iloc[0:215000]), axis = 0)
train = train.drop(['image_crop'], axis=1)
import gc; 

gc.collect()
x_train = x_train.astype('float32')
# Normalise array values



x_train /= 255
# Label is the target variable



y_train = train['label'][0:215000]
del train
import gc; 

gc.collect()
# Neural network variables to be used



img_rows, img_cols = 48, 48



input_shape = (img_rows, img_cols, 3)



batch_size = 128

epochs = 4
# Neural network with multiple layers



model = Sequential()

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(128, activation='sigmoid'))

model.add(Dropout(0.1))

model.add(Dense(1, activation='sigmoid'))
#model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

#model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train model



model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
del x_train
import gc; 

gc.collect()
# Create list of test image files



image_file = []

for file in os.listdir("../input/test/"):

    image_file.append(file)
# Create test data frame



test = pd.DataFrame(image_file,columns=['file'])
test.head()
# Function to generate image test file



def test_func_image_file(x):

    folder = '../input/test/'

    path = folder + x

    return path
test['path'] = test['file'].apply(test_func_image_file)
# Test data image processing



test['image'] = test['path'][0:].map(imread)
test['image_crop'] = test['image'][0:].map(crop)
test = test.drop(['image'], axis=1)
x_test = np.stack(list(test.image_crop.iloc[0:]), axis = 0)
test = test.drop(['image_crop'], axis=1)
import gc; 

gc.collect()
x_test = x_test.astype('float32')
x_test /= 255
test['id'] = test['file'].apply(lambda x: os.path.splitext(x)[0])
predictions = model.predict(x_test)
predictions = predictions.reshape(len(x_test),)
predictions = (predictions > 0.5).astype(np.int)
test['label'] = pd.Series(predictions)
print("Cancer Detected - True Positive --> ",len(test['label'][test['label']==1]))
print("NO Cancer Detected - True Negative --> ",len(test['label'][test['label']==0]))
test = test.drop(['file','path'], axis=1)
test.head()
test.to_csv("submission.csv", columns = test.columns, index=False)