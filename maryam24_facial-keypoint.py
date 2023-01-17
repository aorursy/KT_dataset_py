# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# !pip install tensorflow

# !pip install Theano==0.9

import os

import tensorflow as tf

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from IPython.display import clear_output

from time import sleep

from sklearn.model_selection import train_test_split

# from nolearn.lasagne import BatchIterator

import functools, datetime



Id = pd.read_csv("/kaggle/input/fpoints/IdLookupTable.csv")

train = pd.read_csv('/kaggle/input/fpoints/training.csv')

SampleSubmission = pd.read_csv("/kaggle/input/fpoints/SampleSubmission.csv")

test = pd.read_csv('/kaggle/input/fpoints/test.csv')
train.fillna(method='ffill', inplace = True)
train.head(10)
image = np.array(train['Image'][0].split(' '), dtype = np.float32).reshape(96, 96)
y_train = train.drop('Image', axis = 1)

t = y_train.iloc[0].values

plt.imshow(image, cmap='gray')

plt.scatter(t[0::2], t[1::2], c='yellow', marker='o')
temp = train['Image']

labels = train.drop('Image', axis = 1)
X = []

for i in range(len(temp)):

    img = np.array(temp[i].split(' '), dtype='float').reshape(96,96,1)

    X.append(img)

X = np.array(X)





















# def process(X, y=None):

#     imgs = [np.array(i.split(' '), dtype=np.float32).reshape(96, 96, 1) for i in X]

#     imgs = [img / 255.0 for img in imgs]

#     return np.array(imgs), y
# X_train, X_test, y_train, y_test = train_test_split(train, y_train, random_state = 42, test_size = 0.3)



# X_train, y_train = process(X_train, y_train.values)

# print(X_train.shape)

# print(y_train.shape)



X_train, X_test, y_train, y_test = train_test_split(X, labels, random_state = 42, test_size = 0.3)
activations = 'relu'

inputs_img = tf.keras.layers.Input

convolutional = functools.partial(tf.keras.layers.Conv2D, activation = activations, padding = 'same')



batch_norm = tf.keras.layers.BatchNormalization

max_pool = tf.keras.layers.MaxPool2D

fc =functools.partial(tf.keras.layers.Dense)



flat = tf.keras.layers.Flatten



def model():

    input = inputs_img(shape=(96,96,1))

    conv1 = convolutional(32,(1, 1))(input)

    batch1 = batch_norm()(conv1)

    max_pool1 = max_pool((2,2))(batch1)

    

    conv2 = convolutional(16, (2, 2))(max_pool1)

    batch2 = batch_norm()(conv2)

    max_pool2 = max_pool((3,3))(batch2)

    

    flat1 = flat()(max_pool2)

    

    fc1 = fc(64)(flat1)

    fc2 = fc(30)(fc1)

    

    fullmodel = tf.keras.Model(input, fc2)

    fullmodel.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'acc'])

    

    return fullmodel

model1 = model()

model1.summary()
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(filepath='./weights.hdf5', verbose=1, save_best_only=True),

    tensorboard_callback,

    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

]

    
history = model1.fit(X_train, y_train, epochs = 500, batch_size = 64, validation_split = 0.2)

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
score = model1.evaluate(X_test, y_test, verbose = 0)

print("Test Loss : ", score[0])

print("Test accuracy : ", score[1])
preds = model1.predict(X_train[-1000:], verbose=1)
from skimage.io import imshow

ix = 8 #Type a number between 0 and 999 inclusive

imshow(np.squeeze(X_train[4934-(1000-ix),:,:,0:3]).astype(float)*255) #Only seeing the RGB channels

plt.show()

#Tells what the image is

print ('Prediction:\n{:.1f}% probability barren land,\n{:.1f}% probability trees,\n{:.1f}% probability grassland,\n{:.1f}% probability other\n'.format(preds[ix,0]*100,preds[ix,1]*100,preds[ix,2]*100,preds[ix,3]*100))



print ('Ground Truth: ',end='')

if y_train[4934-(1000-ix),0] == 1:

    print ('Barren Land')

elif y_train[4934-(1000-ix),1] == 1:

    print ('Trees')

elif y_train[4934-(1000-ix),2] == 1:

    print ('Grassland')

else:

    print ('Other')
model1.save('keypoint_model22.h5')
test['Image'] = test['Image'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape((96,96)))
test_X = np.asarray([test['Image']], dtype=np.uint8).reshape(test.shape[0],96,96,1)

test_res = model1.predict(test_X)
n = 22



xv = X_train[n].reshape((96,96))

plt.imshow(xv,cmap='gray')



for i in range(1,31,2):

#     plt.plot(train_predicts[n][i-1], train_predicts[n][i], 'ro')

    plt.plot(y_train[n][i-1], y_train[n][i], 'x', color='green')



plt.show()