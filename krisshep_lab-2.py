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
import numpy as np

import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow import keras
mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print (train_images.shape)

print (train_labels.shape)

print (test_images.shape)

print (test_labels.shape)
train_images = train_images/255.0

test_images = test_images/255.0
plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.imshow(train_images[i])

plt.show()
model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28,28)),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(10, activation='softmax')

])
model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)

from sklearn import datasets as skd



wine = skd.load_wine()

brcancer = skd.load_breast_cancer()
#[print(min(wine.data[:,i])) for i in range(0,13)]

#print('\n')

#[print(max(wine.data[:,i])) for i in range(0,13)]



wine_data_new = np.zeros(wine.data.shape)

for i in range(0,wine.data.shape[1]):

    wrow = wine.data[:,i]

    wine_data_new[:, i] = (wrow-min(wrow))/(max(wrow)-min(wrow))

print(wine_data_new[0][:])

print(wine.data[0][:])

shuffle_ind = np.arange(0, wine_data_new.shape[0])

np.random.shuffle(shuffle_ind)

print(wine_data_new[0][:])

wine_data_shuffle = wine_data_new[shuffle_ind]

print(wine_data_shuffle[0][:])
#print(shuffle_ind)

shuffle_targets = wine.target[shuffle_ind]
wmodel = keras.Sequential([

    keras.layers.Dense(128, input_shape = (13,), activation='relu'),

    keras.layers.Dense(3, activation='softmax')

])

wmodel.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])
w_hist = wmodel.fit(wine_data_shuffle[:133,:], shuffle_targets[:133], epochs=30)
w_test_loss, w_test_acc = wmodel.evaluate(wine_data_shuffle[133:], shuffle_targets[133:], verbose = 1)
wmodel_0 = keras.Sequential([

    keras.layers.Dense(128, input_shape = (13,), activation='tanh'),

    keras.layers.Dense(3, activation='softmax')

])

wmodel_0.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])



wmodel_1 = keras.Sequential([

    keras.layers.Dense(128, input_shape = (13,), activation='sigmoid'),

    keras.layers.Dense(3, activation='softmax')

])

wmodel_1.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])



wmodel_2 = keras.Sequential([

    keras.layers.Dense(128, input_shape = (13,), activation='selu'),

    keras.layers.Dense(3, activation='softmax')

])

wmodel_2.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])
w_hist_0 = wmodel_0.fit(wine_data_shuffle[:133,:], shuffle_targets[:133], epochs=30)

w_hist_1 = wmodel_1.fit(wine_data_shuffle[:133,:], shuffle_targets[:133], epochs=30)

w_hist_2 = wmodel_2.fit(wine_data_shuffle[:133,:], shuffle_targets[:133], epochs=30, batch_size=10)

wmodel_0.evaluate(wine_data_shuffle[133:], shuffle_targets[133:], verbose = 1)

wmodel_1.evaluate(wine_data_shuffle[133:], shuffle_targets[133:], verbose = 1)

wmodel_2.evaluate(wine_data_shuffle[133:], shuffle_targets[133:], verbose = 1, batch_size=10)
plt.figure(figsize=plt.figaspect(0.5))

plt.subplot(1,2,1)

l = range(0, len(w_hist.history['accuracy']))

plt.plot(l, w_hist.history['accuracy'])

plt.plot(l, w_hist_0.history['accuracy'])

plt.plot(l, w_hist_1.history['accuracy'])

plt.plot(l, w_hist_2.history['accuracy'])

plt.legend(['ReLU', 'Tahn', 'Sigmoid', 'SeLU'])

plt.title('Accuracies')



plt.subplot(1,2,2)

l = range(0, len(w_hist.history['loss']))

plt.plot(l, w_hist.history['loss'])

plt.plot(l, w_hist_0.history['loss'])

plt.plot(l, w_hist_1.history['loss'])

plt.plot(l, w_hist_2.history['loss'])

plt.legend(['ReLU', 'Tahn', 'Sigmoid', 'SeLU'])

plt.title('Losses')

plt.show()



wmodel_3 = keras.Sequential([

    keras.layers.Dense(128, input_shape = (13,), activation='relu'),

    keras.layers.Dense(3, activation='softmax')

])

wmodel_3.compile(optimizer='sgd',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])



wmodel_4 = keras.Sequential([

    keras.layers.Dense(128, input_shape = (13,), activation='relu'),

    keras.layers.Dense(3, activation='softmax')

])

optim = keras.optimizers.SGD(momentum=0.3)

wmodel_4.compile(optimizer=optim,

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])



wmodel_5 = keras.Sequential([

    keras.layers.Dense(128, input_shape = (13,), activation='relu'),

    keras.layers.Dense(3, activation='softmax')

])

wmodel_5.compile(optimizer='sgd',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])
w_hist_3 = wmodel_3.fit(wine_data_shuffle[:133,:], shuffle_targets[:133], epochs=30)

w_hist_4 = wmodel_4.fit(wine_data_shuffle[:133,:], shuffle_targets[:133], epochs=30)

w_hist_5 = wmodel_5.fit(wine_data_shuffle[:133,:], shuffle_targets[:133], epochs=30, batch_size=10)

wmodel_3.evaluate(wine_data_shuffle[133:], shuffle_targets[133:], verbose = 1)

wmodel_4.evaluate(wine_data_shuffle[133:], shuffle_targets[133:], verbose = 1)

wmodel_5.evaluate(wine_data_shuffle[133:], shuffle_targets[133:], verbose = 1, batch_size=10)
plt.figure(figsize=plt.figaspect(0.5))

plt.subplot(1,2,1)

l = range(0, len(w_hist.history['accuracy']))

plt.plot(l, w_hist.history['accuracy'])

plt.plot(l, w_hist_3.history['accuracy'])

plt.plot(l, w_hist_4.history['accuracy'])

plt.plot(l, w_hist_5.history['accuracy'])

plt.legend(['Adam', 'SGD', 'SGD+m', 'mb-SGD'])

plt.title('Accuracies')



plt.subplot(1,2,2)

l = range(0, len(w_hist.history['loss']))

plt.plot(l, w_hist.history['loss'])

plt.plot(l, w_hist_3.history['loss'])

plt.plot(l, w_hist_4.history['loss'])

plt.plot(l, w_hist_5.history['loss'])

plt.legend(['Adam', 'SGD', 'SGD+m', 'mb-SGD'])

plt.title('Losses')

plt.show()
print(brcancer.data.shape)

print(brcancer.target.shape)



brcancer_data_new = np.zeros(brcancer.data.shape)

for i in range(0,brcancer.data.shape[1]):

    brow = brcancer.data[:,i]

    brcancer_data_new[:, i] = (brow-min(brow))/(max(brow)-min(brow))



shuffle_ind_b = np.arange(0, brcancer_data_new.shape[0])

np.random.shuffle(shuffle_ind_b)

print(brcancer_data_new[0][:])

brcancer_data_shuffle = brcancer_data_new[shuffle_ind_b]

print(brcancer_data_shuffle[0][:])



b_targets_shuffle = brcancer.target[shuffle_ind_b]
bmodel = keras.Sequential([

    keras.layers.Dense(128, input_shape = (30,), activation='relu'),

    keras.layers.Dense(2, activation='softmax')

])

bmodel.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])
thr_ind = (int)(0.75*brcancer_data_shuffle.shape[0])

b_hist = bmodel.fit(brcancer_data_shuffle[:thr_ind], b_targets_shuffle[:thr_ind], epochs=15)
b_test_loss, b_test_acc = bmodel.evaluate(brcancer_data_shuffle[thr_ind:], b_targets_shuffle[thr_ind:], verbose = 1)
bmodel_0 = keras.Sequential([

    keras.layers.Dense(128, input_shape = (30,), activation='relu'),

    keras.layers.Dense(64, activation='relu'),

    keras.layers.Dense(2, activation='softmax')

])

bmodel_0.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])



bmodel_1 = keras.Sequential([

    keras.layers.Dense(64, input_shape = (30,), activation='relu'),    

    keras.layers.Dense(64, activation='relu'),

    keras.layers.Dense(2, activation='softmax')

])

bmodel_1.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])



bmodel_2 = keras.Sequential([

    keras.layers.Dense(64, input_shape = (30,), activation='relu'),

    keras.layers.Dense(32, activation='relu'),    

    keras.layers.Dense(32, activation='relu'),

    keras.layers.Dense(2, activation='softmax')

])

bmodel_2.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])
b_hist_0 = bmodel_0.fit(brcancer_data_shuffle[:thr_ind], b_targets_shuffle[:thr_ind], epochs=15)

b_hist_1 = bmodel_1.fit(brcancer_data_shuffle[:thr_ind], b_targets_shuffle[:thr_ind], epochs=15)

b_hist_2 = bmodel_2.fit(brcancer_data_shuffle[:thr_ind], b_targets_shuffle[:thr_ind], epochs=15)



b_test_loss, b_test_acc = bmodel_0.evaluate(brcancer_data_shuffle[thr_ind:], b_targets_shuffle[thr_ind:], verbose = 1)

b_test_loss, b_test_acc = bmodel_1.evaluate(brcancer_data_shuffle[thr_ind:], b_targets_shuffle[thr_ind:], verbose = 1)

b_test_loss, b_test_acc = bmodel_2.evaluate(brcancer_data_shuffle[thr_ind:], b_targets_shuffle[thr_ind:], verbose = 1)
plt.figure(figsize=plt.figaspect(0.5))

plt.subplot(1,2,1)

l = range(0, len(b_hist.history['accuracy']))

plt.plot(l, b_hist.history['accuracy'])

plt.plot(l, b_hist_0.history['accuracy'])

plt.plot(l, b_hist_1.history['accuracy'])

plt.plot(l, b_hist_2.history['accuracy'])

plt.legend(['128', '128;64', '64;64', '64;32;32'])

plt.title('Accuracies')



plt.subplot(1,2,2)

plt.plot(l, b_hist.history['loss'])

plt.plot(l, b_hist_0.history['loss'])

plt.plot(l, b_hist_1.history['loss'])

plt.plot(l, b_hist_2.history['loss'])

plt.legend(['128', '128;64', '64;64', '64;32;32'])

plt.title('Losses')

plt.show()