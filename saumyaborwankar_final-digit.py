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
import matplotlib.pyplot as plt

import seaborn as sns

import keras

from keras.models import Sequential

from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix
train_df = pd.read_csv("../input/digit-recognizer/train.csv")

test_df = pd.read_csv("../input/digit-recognizer/test.csv")

submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
y_train = train_df['label']

y = train_df['label']

del train_df['label']

#print(train_df.shape,y_train.shape)
print(train_df.shape,y_train.shape)
from sklearn.preprocessing import LabelBinarizer

label_binarizer = LabelBinarizer()

y_train = label_binarizer.fit_transform(y_train)

print(train_df.shape,y_train.shape)

x_train = train_df.values

x_test = test_df.values

print(x_train.shape,x_test.shape)
# Normalize the data

x_train = x_train / 255

x_test = x_test / 255
x_train = x_train.reshape(-1,28,28,1)

x_test = x_test.reshape(-1,28,28,1)

print(x_train.shape,x_test.shape,y_train.shape)
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(x_train)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size = 0.1)

print(x_train.shape,x_val.shape,y_train.shape,y_val.shape)
model = Sequential()

model.add(Conv2D(150 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(100 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))

model.add(Dropout(0.1))

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(75 , (5,5) , strides = 1 , padding = 'same' , activation = 'relu'))

model.add(Dropout(0.2))

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(50 , (4,4) , strides = 1 , padding = 'same' , activation = 'relu'))

model.add(Dropout(0.15))

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(25 , (4,4) , strides = 1 , padding = 'same' , activation = 'relu'))

model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Flatten())

model.add(Dense(units = 512 , activation = 'relu'))

model.add(Dropout(0.25))

model.add(Dense(units = 10 , activation = 'softmax'))

model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

model.summary()
history = model.fit(datagen.flow(x_train,y_train, batch_size = 256) ,epochs = 30 , validation_data = (x_val, y_val))
#new_model=tf.keras.models.load_model('digit.h5')

y_pred=model.predict_classes(x_test)

submission.head()

submission=pd.DataFrame({'ImageId': submission.ImageId,'Label':y_pred})

submission.to_csv('/kaggle/working/submission.csv',index=False)

check=pd.read_csv('/kaggle/working/submission.csv')

check