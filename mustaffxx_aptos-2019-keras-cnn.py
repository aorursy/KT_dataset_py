# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import tensorflow as tf

from tensorflow import keras

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
PATH = '../input/'

train = pd.read_csv(PATH + 'train.csv')
train.head()
train['diagnosis'].value_counts().plot(kind = 'bar')
train['diagnosis'].value_counts().sort_index()[0]
total = train['diagnosis'].value_counts().sum()

w0 = train['diagnosis'].value_counts().sort_index()[0] / total

w1 = train['diagnosis'].value_counts().sort_index()[1] / total

w2 = train['diagnosis'].value_counts().sort_index()[2] / total

w3 = train['diagnosis'].value_counts().sort_index()[3] / total

w4 = train['diagnosis'].value_counts().sort_index()[4] / total

class_wg = {0: w3, 1: w1, 2: w4, 3: w0, 4: w2}

#class_wg = {0: w0, 1: w1, 2: w2, 3: w3, 4: w4}



print('class weights:')

print('0: ', w0)

print('1: ', w1)

print('2: ', w2)

print('3: ', w3)

print('4: ', w4)
from keras.preprocessing import image

from keras.applications.imagenet_utils import preprocess_input



img = image.load_img(PATH + 'train_images/' + train['id_code'][0] + '.png', target_size = (100, 100, 3))

plt.imshow(img)
def load_images(df, dfPath):

    xdata = np.zeros((df.shape[0], 100, 100, 3))

    index = 0

    for id_code in df['id_code']:

        img = image.load_img(PATH + dfPath + '/' + id_code + '.png', target_size = (100, 100, 3))

        x = image.img_to_array(img)

        x = preprocess_input(x)

        xdata[index] = x

        index += 1

    xdata = xdata / 255.0

    return xdata       
x_train = train.drop(['diagnosis'], axis = 1)

y_train = train['diagnosis']
trainpath = 'train_images'

x_train = load_images(train, trainpath)
plt.imshow(x_train[0][:,:,:])
plt.imshow(x_train[0][:,:,0])
plt.imshow(x_train[0][:,:,1])
plt.imshow(x_train[0][:,:,2])
y_train.head()
print(y_train.shape)
del train
from sklearn.preprocessing import LabelEncoder

from keras.utils.np_utils import to_categorical



lb = LabelEncoder()

y_train = lb.fit_transform(y_train)

y_train = to_categorical(y_train, num_classes = 5)
y_train
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D



model = Sequential()



model.add(Conv2D(filters = 30, kernel_size = (5, 5), input_shape = (100, 100, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.2))



model.add(Conv2D(filters = 15, kernel_size = (3, 3), activation = 'relu'))

model.add(Conv2D(filters = 15, kernel_size = (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(128, activation = 'relu'))

model.add(Dense(64, activation = 'relu'))

model.add(Dense(32, activation = 'relu'))



model.add(Dense(5, activation = 'softmax'))



model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['categorical_accuracy'])

model.summary()
model.fit(x_train, y_train, epochs = 50, batch_size = 200, class_weight = class_wg)
del x_train, y_train
testPath = 'test_images'

test = pd.read_csv(PATH + 'test.csv')
test.head()
x_test = load_images(test, testPath)
predictions = model.predict(x_test, verbose = 1)
for i in range(len(test)):

    test.loc[i, 'diagnosis'] = np.argmax(predictions[i - 1])

test['diagnosis'] = test['diagnosis'].astype('int')
test.head()
test.to_csv('submission.csv', index = False)