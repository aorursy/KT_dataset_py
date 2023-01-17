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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



train_x_orig = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_x_orig = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



print(train_x_orig.shape)

print(test_x_orig.shape)
print('traiing dataset columns',train_x_orig.head())

print('\n test set columns',test_x_orig.head())



print(type(train_x_orig))

print(type(test_x_orig))
labels_train = (train_x_orig['label'].values).reshape(-1,1)



pixels_train = train_x_orig.drop('label', axis = 1)



print('Shape of label_train ',labels_train.shape)

print('\n Shape of pixels train',pixels_train.shape)



print('\n First 10 values of Labels_train',labels_train[:10])
# 1st to convert DATAFRAME to uint8 datatype as array



im = np.array(pixels_train, dtype = 'uint8')

print('IMAGE DATA TYPE ',type(im))



# taking 2nd image from dataset of 42,000, python 0 = real world 1, python 1 = real world 2



im1 = im[1] # changing [this number] for im will give you any index of image you want to visualize

print('\n shape of 2nd image pixels = ', im1.shape)
im1 = im1.reshape((28,28))

plt.imshow(im1, cmap = 'gray')
print(labels_train[1])
from sklearn.model_selection import train_test_split



pix_train, pix_valid, label_train, label_valid = train_test_split(pixels_train, labels_train, test_size = 0.30)

# data types of all variables

print(' Data type for pix_train = ',type(pix_train))

print(' Data type for label_train = ',type(label_train))

print(' Data type for pix_valid = ',type(pix_valid))

print(' Data type for label_valid = ',type(label_valid))
pix_train = (pix_train.values).astype('float32')

pix_test = (test_x_orig.values).astype('float32')

pix_valid = (pix_valid.values).astype('float32')





print(' Data type for pix_train = ',type(pix_train))

print(' Data type for pix_valid = ',type(pix_valid))
pix_train /= 255.0

pix_test  /= 255.0

pix_valid /= 255.0



print('Maximum value in pix_train =', np.max(pix_test))

print('Minimum value in pix_train =', np.min(pix_train))
print('Shape of pix_train =',pix_train.shape)
# reshaping

pix_train = pix_train.reshape(pix_train.shape[0], 28,28,1)

pix_test = pix_test.reshape(pix_test.shape[0], 28, 28,1)

pix_valid = pix_valid.reshape(pix_valid.shape[0], 28, 28,1)



print('Shape of pix_train after reshaping=',pix_train.shape)
num_classes = len(np.unique(labels_train))



print('Number of classes', num_classes)
# using ONE HOT ENCODER

from keras.utils import to_categorical



label_train = to_categorical(label_train)

label_valid = to_categorical(label_valid)



print('after one hot encoder',label_valid[0])
import tensorflow as tf

from keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D

from keras.models import Sequential



model = Sequential()



model.add(Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'valid', activation = 'relu', input_shape = pix_train.shape[1:] ))

model.add(Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'valid', activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'valid', activation = 'relu'))

model.add(Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'valid', activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Flatten())



model.add(Dense(512, activation = 'relu'))

model.add(Dropout(0.3))

model.add(BatchNormalization())



model.add(Dense(256, activation = 'relu'))

model.add(Dropout(0.3))

model.add(BatchNormalization())



model.add(Dense(num_classes, activation = 'softmax'))



model.compile(loss = 'categorical_crossentropy', optimizer = 'SGD', metrics = ['accuracy'])
model.summary()
hist = model.fit(pix_train, label_train, epochs = 100, batch_size = 64, validation_data = (pix_valid, label_valid))
# plot loss during training

plt.subplot(211)

plt.title('Loss')

plt.plot(hist.history['loss'], label='train')

plt.plot(hist.history['val_loss'], label='test')

plt.legend()

# plot accuracy during training

plt.subplot(212)

plt.title('Accuracy')

plt.plot(hist.history['acc'], label='train')

plt.plot(hist.history['val_acc'], label='test')

plt.legend()

plt.show()
results = model.predict(pix_test)



results = np.argmax(results, axis = 1)



results = pd.Series(results, name = 'Label')



sub = pd.concat([pd.Series(range(1,28001), name = 'ImageId'), results], axis = 1)



sub.to_csv('csv_to_submit.csv', index = False)
