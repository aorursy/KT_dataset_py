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
work_dir = '../input/cifar10batchespy/cifar-10-batches-py/'

work_dir
def unpickle(file):

    import pickle

    with open(file, 'rb') as fo:

        work_dict = pickle.load(fo, encoding = 'bytes')

    return work_dict  
dirs = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
all_data = [0,1,2,3,4,5,6]
for i, direc in zip(all_data, dirs):

    all_data[i] = unpickle(work_dir + direc)
print(all_data[1])
test_batch = all_data[6]

test_batch[b'data'].shape
len(test_batch[b'labels'])
test_batch.keys()
test_batch[b'data'][0].reshape(32,32,3)
import matplotlib.pyplot as plt
plt.imshow(test_batch[b'data'][0].reshape(32,32,3))
plt.imshow(test_batch[b'data'][0][:1024].reshape(32,32))
plt.imshow(test_batch[b'data'][2][:1024].reshape(32,32))
data_r = test_batch[b'data'][:,:1024].reshape(-1,32,32)

data_g = test_batch[b'data'][:,1024:2*1024].reshape(-1,32,32)

data_b = test_batch[b'data'][:,2*1024:].reshape(-1,32,32)
viewable_images = np.stack([data_r, data_g, data_b], axis = -1)
plt.imshow(viewable_images[1])
viewable_images.shape
labels=np.array(test_batch[b'labels'])

labels.shape
from sklearn.linear_model import LogisticRegression
simple_model = LogisticRegression()
#simple_model.fit(test_batch[b'data'],labels)
from sklearn.model_selection import train_test_split
train_image,test_image, train_labels,test_labels = train_test_split(test_batch[b'data'], labels)
test_labels.shape
from keras.layers import Dense

from keras.models import Sequential
simple_model = Sequential()

#simple_model.add(Convo3D(100))(Dense(10,activation = 'softmax',input_shape = (32*32*3,)))

#simple_model.add(Dense(10,activation = 'sigmoid',input_shape = (32*32*3,)))

#simple_model.add(Dense(10,activation = 'softmax'))

simple_model.compile('adam','sparse_categorical_crossentropy' , metrics = ['acc'])

simple_model.add(Dense(10000,bias_initializer = "zeros",activation = "tanh",input_shape = (32*32*3,)))

simple_model.compile('adam','sparse_categorical_crossentropy' , metrics = ['acc'])

simple_model.add(Dense(1000,bias_initializer = "zeros",activation = "relu"))

simple_model.compile('adam','sparse_categorical_crossentropy' , metrics = ['acc'])

simple_model.add(Dense(100,bias_initializer = "zeros",activation = "softmax"))

simple_model.compile('adam','sparse_categorical_crossentropy' , metrics = ['acc'])

simple_model.add(Dense(10,bias_initializer = "zeros",activation = "sigmoid"))

simple_model.compile('adam','sparse_categorical_crossentropy' , metrics = ['acc'])

#simple_model.add(Dropout(0.3))

#simple_model.add(Conv2D(32, (3,3), padding='same',input_shape = (32*32*3,)))

#simple_model.add(Activation('elu'))

#simple_model.add(BatchNormalization())

#simple_model.add(Conv2D(32, (3,3)))

#simple_model.add(Activation('elu'))

#simple_model.add(BatchNormalization())

#simple_model.add(MaxPooling2D(pool_size=(2,2)))

#simple_model.add(Dropout(0.2))
simple_model.fit(train_image, train_labels, validation_data=(test_image, test_labels),epochs = 10)
simple_model.predict(test_image[5:10])
simple_model.predict(test_image[5:10])
plt.imshow(np.stack([test_image[8][:1024].reshape(32,32), 

                     test_image[8][1024:2*1024].reshape(32,32), 

                     test_image[8][2*1024:].reshape(32,32)],axis = -1))
plt.imshow(data_r[0])
plt.imshow(np.stack([data_r,data_g,data_b], axis = -1)[0])
plt.imshow(test_batch[b'data'][1].reshape(32,32,3))
unpickle(work_dir+'batches.meta')[b'label_names']