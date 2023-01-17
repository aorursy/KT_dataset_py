import tflearn

from keras.utils import np_utils

from tflearn.data_utils import shuffle, to_categorical

from tflearn.layers.core import input_data, dropout, fully_connected

from tflearn.layers.conv import conv_2d, max_pool_2d

import tensorflow as tf

from tflearn.layers.estimator import regression

tf.reset_default_graph()

from sklearn.model_selection import train_test_split

import numpy as np

def unpickle(file):

    import pickle

    with open(file, 'rb') as fo:

        dict = pickle.load(fo, encoding='bytes')

    return dict
myDict1 = {"key1": [12, 14], "key2": [1,2,3]}

myDict2 = {"key1": [16, 18], "key2": [11,12,13]}



for key in myDict2:

    print(key, myDict2[key], myDict1[key])

    print(myDict1[key].extend(myDict2[key]))

    

myDict1
myDictTrain = unpickle("..//input//data_batch_1")

myDict2 = unpickle("..//input//data_batch_2")

myDict3 = unpickle("..//input//data_batch_3")

myDict4 = unpickle("..//input//data_batch_4")

myDict5 = unpickle("..//input//data_batch_5")

myDictTest = unpickle("..//input//test_batch")



print(len(myDictTrain[b'labels']), len(myDict2[b'labels']), len(myDict3[b'labels']), len(myDict4[b'labels']), len(myDict4[b'labels']))
myDictTrain[b'labels'].extend(myDict2[b'labels'])

myDictTrain[b'labels'].extend(myDict3[b'labels'])

myDictTrain[b'labels'].extend(myDict4[b'labels'])

myDictTrain[b'labels'].extend(myDict5[b'labels'])

len(myDictTrain[b'labels'])

# myDictTrain[b'data'] = np.concatenate(myDictTrain[b'data'], myDict2[b'data'])



print(type(myDictTrain[b'data']))

print(type(myDict2[b'data']))



print("Length:", myDictTrain[b'data'].shape, myDict2[b'data'].shape)



myDictTrain[b'data'] = np.concatenate((myDictTrain[b'data'], myDict2[b'data'],  myDict3[b'data'],  myDict4[b'data'],  myDict5[b'data']), axis=0)

print(type(myDictTrain[b'data']))

print(myDictTrain[b'data'].shape)
for key in myDictTrain:

    print(key)
y_train = myDictTrain[b'labels']

y_test = myDictTest[b'labels']
X_train = myDictTrain[b'data']

X_test = myDictTest[b'data']
np.asarray(X_train)

X_train.shape
X_train = X_train.reshape(X_train.shape[0], 3, 32, 32).astype('float32')

X_test = X_test.reshape(X_test.shape[0], 3, 32, 32).astype('float32')

# y_train = np_utils.to_categorical(y_train)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 324)
X_train = X_train - np.mean(X_train) / X_train.std()

X_test = X_test - np.mean(X_test) / X_test.std()
# y_train = np_utils.to_categorical(y_train)

# y_test = np_utils.to_categorical(y_test)

num_classes = len(set(y_test))
num_classes