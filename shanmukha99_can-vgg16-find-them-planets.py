import numpy as np

import pandas as pd



##thanks to https://www.kaggle.com/toregil/mystery-planet-99-8-cnn and https://www.kaggle.com/aleksod/0-75-precision-0-60-recall-linear-svc

##very useful kernels pls have a look at their implementations too!
train = pd.read_csv("/kaggle/input/kepler-labelled-time-series-data/exoTrain.csv")
test = pd.read_csv("/kaggle/input/kepler-labelled-time-series-data/exoTest.csv")
train.head()
train.groupby('LABEL').size()
train.describe()
test.groupby('LABEL').size()
train[0:37]
import matplotlib.pyplot as plt
for i in range(0,10): #random numbers :/

    flux = train[train.LABEL == 1].drop('LABEL', axis=1).iloc[i,:]

    time = np.arange(len(flux)) * (36.0/60.0) # time in units of hours

    plt.figure(figsize=(15,5))

    plt.title('Flux of star {} with  confirmed exoplanets'.format(i+1),color='r')

    plt.ylabel('Flux, e-/s',color='r')

    plt.xlabel('Time, hours',color='r')

    plt.plot(time, flux)
for i in range(10,20): #random numbers :/

    flux = train[train.LABEL == 1].drop('LABEL', axis=1).iloc[i,:]

    time = np.arange(len(flux)) * (36.0/60.0) # time in units of hours

    plt.figure(figsize=(15,5))

    plt.title('Flux of star {} with  confirmed exoplanets'.format(i+1),color='r')

    plt.ylabel('Flux, e-/s',color='r')

    plt.xlabel('Time, hours',color='r')

    plt.plot(time, flux)
import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPool1D, Flatten

from keras import backend as K

from keras.preprocessing import image

from keras.applications.mobilenet import MobileNet

from keras.applications.vgg16 import preprocess_input, decode_predictions

from keras.models import Model

import timeit

import warnings

warnings.filterwarnings('ignore')
INPUT_LIB = '/kaggle/input/kepler-labelled-time-series-data/'

raw_data = np.loadtxt(INPUT_LIB + 'exoTrain.csv', skiprows=1, delimiter=',')

x_train = raw_data[:, 1:]

y_train = raw_data[:, 0, np.newaxis] - 1.

raw_data = np.loadtxt(INPUT_LIB + 'exoTest.csv', skiprows=1, delimiter=',')

x_test = raw_data[:, 1:]

y_test = raw_data[:, 0, np.newaxis] - 1.

del raw_data
x_train = ((x_train - np.mean(x_train, axis=1).reshape(-1,1)) / 

           np.std(x_train, axis=1).reshape(-1,1))

x_test = ((x_test - np.mean(x_test, axis=1).reshape(-1,1)) / 

          np.std(x_test, axis=1).reshape(-1,1))
from scipy.ndimage.filters import uniform_filter1d

x_train = np.stack([x_train, uniform_filter1d(x_train, axis=1, size=200)], axis=2)

x_test = np.stack([x_test, uniform_filter1d(x_test, axis=1, size=200)], axis=2)
x_train = x_train - x_train.mean()

x_train = x_train / x_train.max()
x_test = x_test - x_test.mean()

x_test = x_test / x_test.max()
model = Sequential()

model.add(Conv1D(input_shape=x_train.shape[1:],filters=64,kernel_size=9,padding="same", activation="relu"))

model.add(Conv1D(filters=64,kernel_size=9,padding="same", activation="relu"))

model.add(MaxPool1D(pool_size=4,strides=4))

model.add(Conv1D(filters=128, kernel_size=9, padding="same", activation="relu"))

model.add(Conv1D(filters=128, kernel_size=9, padding="same", activation="relu"))

model.add(MaxPool1D(pool_size=4,strides=4))

model.add(Conv1D(filters=256, kernel_size=9, padding="same", activation="relu"))

model.add(Conv1D(filters=256, kernel_size=9, padding="same", activation="relu"))

model.add(Conv1D(filters=256, kernel_size=9, padding="same", activation="relu"))

model.add(MaxPool1D(pool_size=4,strides=4))

model.add(Conv1D(filters=512, kernel_size=9, padding="same", activation="relu"))

model.add(Conv1D(filters=512, kernel_size=9, padding="same", activation="relu"))

model.add(Conv1D(filters=512, kernel_size=9, padding="same", activation="relu"))

model.add(MaxPool1D(pool_size=4,strides=4))

model.add(Conv1D(filters=512, kernel_size=9, padding="same", activation="relu"))

model.add(Conv1D(filters=512, kernel_size=9, padding="same", activation="relu"))

model.add(Conv1D(filters=512, kernel_size=9, padding="same", activation="relu"))

model.add(MaxPool1D(pool_size=4,strides=4))



model.add(Flatten())

model.add(Dense(units=4096,activation="relu"))

model.add(Dense(units=4096,activation="relu"))

model.add(Dense(units=1, activation='sigmoid'))
def batch_generator(x_train, y_train, batch_size=32):

    """

    Gives equal number of positive and negative samples, and rotates them randomly in time

    """

    half_batch = batch_size // 2

    x_batch = np.empty((batch_size, x_train.shape[1], x_train.shape[2]), dtype='float32')

    y_batch = np.empty((batch_size, y_train.shape[1]), dtype='float32')

    

    yes_idx = np.where(y_train[:,0] == 1.)[0]

    non_idx = np.where(y_train[:,0] == 0.)[0]

    

    while True:

        np.random.shuffle(yes_idx)

        np.random.shuffle(non_idx)

    

        x_batch[:half_batch] = x_train[yes_idx[:half_batch]]

        x_batch[half_batch:] = x_train[non_idx[half_batch:batch_size]]

        y_batch[:half_batch] = y_train[yes_idx[:half_batch]]

        y_batch[half_batch:] = y_train[non_idx[half_batch:batch_size]]

    

        for i in range(batch_size):

            sz = np.random.randint(x_batch.shape[1])

            x_batch[i] = np.roll(x_batch[i], sz, axis = 0)

     

        yield x_batch, y_batch
from keras.optimizers import Adam

opt = Adam(lr=4e-5)

model.compile(optimizer=opt, loss=keras.losses.binary_crossentropy, metrics=['binary_accuracy'])

hist = model.fit_generator(batch_generator(x_train, y_train, 32), 

                           validation_data=(x_test, y_test), 

                           verbose=2, epochs=50,

                           steps_per_epoch=x_train.shape[1]//32)
plt.plot(hist.history['loss'], color='b')

plt.plot(hist.history['val_loss'], color='r')

plt.show()

plt.plot(hist.history['binary_accuracy'], color='b')

plt.plot(hist.history['val_binary_accuracy'], color='r')

plt.show()
non_idx = np.where(y_test[:,0] == 0.)[0]

yes_idx = np.where(y_test[:,0] == 1.)[0]

y_hat = model.predict(x_test)[:,0]
X = range(len(y_hat))
y_test_pred=[]

for i in y_hat:

    if i>=0.7:

        y_test_pred.append(1)

    else:

        y_test_pred.append(0)
plt.scatter(X,y_test_pred,color='b')

plt.show()

plt.scatter(X,y_test,color='r')

plt.show()
y_test_pred=np.asarray(y_test_pred)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("accuracy ",accuracy_score(y_test, y_test_pred))

print("precision ",precision_score(y_test, y_test_pred))

print("recall-score ",recall_score(y_test, y_test_pred))

print("f1-score ",f1_score(y_test, y_test_pred))        