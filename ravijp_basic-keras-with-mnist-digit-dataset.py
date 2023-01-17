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
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')

print('Size of train is {} and size of test data is {}.'.format(train.shape, test.shape))
x_train = (train.iloc[:,1:].values).astype('float32') # all pixel values
y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits
X_test = test.values.astype('float32')
#Data visualization
X_train = x_train.reshape(x_train.shape[0], 28, 28)
import matplotlib.pyplot as plt
for i in range(6, 9):
    plt.subplot(330 + (i+1))
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    plt.title(y_train[i]);
mean_vals = np.mean(x_train, axis=0)
std_val = np.std(x_train)
x_train_centered = (x_train - mean_vals)/std_val
#x_train_centered = x_train
x_test_centered = (test - mean_vals)/std_val
#x_test_centered = X_test
del train, test, x_train

print(x_train_centered.shape, x_test_centered.shape)
import tensorflow as tf
import tensorflow.contrib.keras as keras
np.random.seed(123)
tf.set_random_seed(123)
#we need to convert class labels 0-9 into one-hot format.
y_train_onehot = keras.utils.to_categorical(y_train)
print('First three y labels: ', y_train[:3])
print('First three y one-hot labels: ', y_train_onehot[:3])
model = keras.models.Sequential()
#1. Input layer 
    #we have to make sure that input_dim attribute matches the number of features in the training set
model.add(
    keras.layers.Dense(
        units=50,
        input_dim=x_train_centered.shape[1],
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='tanh'))
#Hidden layer
model.add(
    keras.layers.Dense(
    units=50, 
    input_dim=50,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    activation='tanh'))

#Output layer
    #number of units in output layer should be equal to number of unique class labels
model.add(
    keras.layers.Dense(
    units=y_train_onehot.shape[1],
    input_dim=50,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    activation='softmax'))
sgd_optimizer = keras.optimizers.SGD(decay=1e-7, lr=.001, momentum=.9)
model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy')
history = model.fit(x_train_centered, y_train_onehot, batch_size=64, epochs=50, verbose=1, validation_split=.15)
y_train_pred = model.predict_classes(x_train_centered, verbose=0)
print('First 3 predictions:', y_train_pred[:3])
correct_preds = np.sum(y_train==y_train_pred, axis=0)
train_acc = correct_preds/len(y_train)
print('Training accuracy :%.2f%%'%(train_acc*100))



y_test_pred = model.predict_classes(x_test_centered, verbose=0)
output = pd.DataFrame({'ImageId': np.arange(len(y_test_pred))+1,
                       'Label': y_test_pred})

output.to_csv('submission.csv', index=False)
output.head()