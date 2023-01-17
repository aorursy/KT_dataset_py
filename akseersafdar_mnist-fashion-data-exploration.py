#making the imports

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# TensorFlow and tf.keras

import tensorflow as tf

from tensorflow import keras

import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt



print(tf.__version__)



#reading the train file

train = pd.read_csv('../input/fashion-mnist_train.csv')
train.head()
#train shape

train.shape
#reading the test file

test = pd.read_csv('../input/fashion-mnist_test.csv')

test.head()
#shape of test data

test.shape
#defining the list for labels

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#splitting the data into X and y

train_images = train.iloc[:,1:785]

train_labels = train.iloc[:,0]



test_images = test.iloc[:,1:785]

test_labels = test.iloc[:,0]
#plotting some fashion items from dataset. 

plt.figure()

plt.imshow(train_images.iloc[0].as_matrix().reshape(28,28))

plt.colorbar()

plt.xticks([])

plt.yticks([])

plt.show()



print(class_names[0])
#plotting some fashion items from dataset. 

plt.figure()

plt.imshow(train_images.iloc[4].as_matrix().reshape(28,28))

plt.colorbar()

plt.xticks([])

plt.yticks([])

plt.show()



print(class_names[3])
#scaling the data so that the values are between 0 and 1



train_images = train_images / 255.0



test_images = test_images / 255.0
#displaying the first 25 images in training set

plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_images.iloc[i].as_matrix().reshape(28,28))

    plt.xlabel(class_names[train_labels[i]])

plt.show()
#building the model



model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation= tf.nn.relu))

model.add(tf.keras.layers.Dense(128, activation= tf.nn.relu))

model.add(tf.keras.layers.Dense(64, activation= tf.nn.relu))

model.add(tf.keras.layers.Dense(10, activation= tf.nn.softmax))
#compile the model



model.compile(loss= 'sparse_categorical_crossentropy', optimizer= 'adam', metrics= ['accuracy'])

#converting to a np array

train_images = train_images.values

train_labels = train_labels.values



test_images = test_images.values

test_labels = test_labels.values



#train the model



model.fit(train_images, train_labels, epochs= 10)
#evaluate the model

test_loss, test_acc = model.evaluate(test_images, test_labels)



print('The test accuracy is: {} and test loss is: {}'.format(test_acc, test_loss))
#getting the predictions



predictions = model.predict(test_images)
#getting the prediction for first row in test set

np.argmax(predictions[0])
#comparing with actual label

print(test_labels[0])

print(class_names[0])
#plot the figure for first element in test set

plt.figure()

plt.imshow(test_images[0].reshape(28,28))

plt.colorbar()

plt.xticks([])

plt.yticks([])

plt.show()
#lets randomly do it for postion 99

x = np.argmax(predictions[999])

print(x)

print('\n')

print(test_labels[999])
#plot the figure for first element in test set

plt.figure()

plt.imshow(test_images[999].reshape(28,28))

plt.colorbar()

plt.xticks([])

plt.yticks([])

plt.show()



print(class_names[x])