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
dataset=pd.read_csv("../input/labels.csv")
dataset_test=pd.read_csv("../input/sample_submission.csv")
target=pd.Series(dataset['breed'])
one_hot=pd.get_dummies(target,sparse=True)
target=np.asarray(one_hot)
im_size= 64#----> image size
ids = dataset['id']
x_train =[]
y_train = target
import cv2
for id in ids:
    img = cv2.imread('../input/train/{}'.format(id)+'.jpg')
    if img is not None:
        
        
        img = cv2.resize(img, (64,64))
        x_train.append(img)
        
    else:
        print("image not loaded {}".format(id))
x_train=np.array(x_train,np.float32)
x_test = list()
ids_test = dataset_test['id']
for id in ids_test:
    img = cv2.imread('../input/test/{}'.format(id)+'.jpg')
    x_test.append(cv2.resize(img,(64,64)))
x_test = np.array(x_test,np.float32)
def standardize(array):
    array/=255
    return array
x_train = standardize(x_train)
x_test = standardize(x_test)

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

# Adding a third convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 120, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

# fits the model on batches with real-time data augmentation:
classifier.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=20)

test_predictions=classifier.predict(x_test,batch_size=32,verbose=1)
dog_species=dataset_test.columns[1:]
submission_rishi=pd.DataFrame(data=test_predictions,index=ids_test,columns=dog_species)
submission_rishi.index.name='id'
submission_rishi.to_csv('submission_rishi.csv',encoding='utf-8',index=True)
