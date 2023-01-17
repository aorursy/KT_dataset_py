# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os



# Python ≥3.5 is required

import sys

assert sys.version_info >= (3, 5)



# Scikit-Learn ≥0.20 is required

import sklearn

assert sklearn.__version__ >= "0.20"

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import MinMaxScaler



# TensorFlow ≥2.0 is required

import tensorflow as tf

from tensorflow import keras

assert tf.__version__ >= "2.0"

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 



# to make this notebook's output stable across runs

np.random.seed(42)

tf.random.set_seed(42)



# To plot pretty figures

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# read train and test dataset

train_df = pd.read_csv("/kaggle/input/facial-keypoints-detection/training.zip")

test_df = pd.read_csv("/kaggle/input/facial-keypoints-detection/test.zip")
train_df.head().T
test_df.head().T
print(train_df.shape)

print(test_df.shape)
# check the train dataset information. You see there are lots of missing label values

train_df.info()
train_df.describe().loc['count'].plot.bar()
# there is 4909 examples from training set with missing values

train_df.T.isnull().any().value_counts()
# we have 2 options: removing any example with missing value | filling the missed values with keypoints average

# removing will decrease our dataset | Filling will produce random error, but it can act as a regularizer



#train_df = train_df.dropna()

train_df.fillna(train_df.drop(labels=['Image'], axis=1).mean(), inplace=True)

train_df.T.isnull().any().value_counts()
# the images are stored in the last column in string format.



X_train_full = np.vstack(train_df['Image'].apply(lambda img: np.fromstring(img, dtype=int, sep=' ')))

X_test = np.vstack(test_df['Image'].apply(lambda img: np.fromstring(img, sep=' ')))

Y_train_full = train_df.drop(labels=['Image'], axis=1)
# it is a good habit to delete unnecessary variables to free up some space

del train_df, test_df
print(X_train_full.shape)

print(X_test.shape)

print(Y_train_full.shape)
#reshape, convert to float32, and normalize the input and output

X_train_full = X_train_full.reshape(-1, 96, 96, 1).astype('float32') / 255.0

X_test = X_test.reshape(-1, 96, 96, 1).astype('float32') / 255.0

Y_train_full = Y_train_full.values.astype('float32')



output_pipe = make_pipeline(

                MinMaxScaler(feature_range=(-1,1))

                )

Y_train_full = output_pipe.fit_transform(Y_train_full)



X_train, x_val, Y_train, y_val = train_test_split(X_train_full, Y_train_full,

                                                  test_size=0.2, random_state=42)
print(X_train.shape)

print(x_val.shape)

print(Y_train.shape)

print(y_val.shape)

print(X_test.shape)

# plot random samples with keypoints from training

def plot_img_with_keypoints(nrows=4, ncols=4):

    selection = np.random.choice(len(X_train), size=(nrows*ncols), replace=False)

    images = X_train[selection]

    keypoints = output_pipe.inverse_transform(Y_train[selection])

    fig, axes = plt.subplots(figsize=(nrows*2, ncols*2), nrows=nrows, ncols=ncols)

    for img, keypoint, ax in zip(images, keypoints, axes.ravel()):

        keypoint = keypoint.reshape(15,2)

        ax.imshow(img.reshape(96,96), cmap='gray')

        ax.scatter(keypoint[:,0], keypoint[:,1], marker='o', s=15)

        ax.axis('off')
plot_img_with_keypoints(4,4)
# CNN model architecture (all these parameters might be tuned to achieve better results)

from functools import partial



DefaultConv2D = partial(Conv2D, activation='relu', padding='SAME')



model = Sequential([

    # input layer

    BatchNormalization(input_shape=(96, 96, 1)),

    DefaultConv2D(24, (5, 5), kernel_initializer='he_normal'),

    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    Dropout(0.2),

    # layer 2

    DefaultConv2D(36, (5, 5)),

    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    Dropout(0.2),

    # layer 3

    DefaultConv2D(48, (5, 5)),

    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    Dropout(0.2),

    # layer 4

    DefaultConv2D(64, (3, 3)),

    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    Dropout(0.2),

    # layer 5

    DefaultConv2D(64, (3, 3)),

    Flatten(),

    # layer 6

    Dense(500, activation="relu"),

    # layer 7

    Dense(90, activation="relu"),

    # layer 8

    Dense(30),

])



# this model acheive much worse RMSE. Do you know why?

'''

model = Sequential([

    BatchNormalization(input_shape=(96, 96, 1)),

    DefaultConv2D(filters=24, kernel_size=7, kernel_initializer='he_normal'),

    BatchNormalization(),

    MaxPooling2D(pool_size=2),

    DefaultConv2D(filters=36, kernel_size=5),

    BatchNormalization(),

    DefaultConv2D(filters=48, kernel_size=5),

    BatchNormalization(),

    MaxPooling2D(pool_size=2),

    DefaultConv2D(filters=64, kernel_size=3),

    BatchNormalization(),

    DefaultConv2D(filters=64, kernel_size=3),

    BatchNormalization(),

    MaxPooling2D(pool_size=2),

    DefaultConv2D(filters=128, kernel_size=3),

    BatchNormalization(),

    MaxPooling2D(pool_size=2),

    Flatten(),

    Dense(units=500, activation='relu'),

    Dropout(0.2),

    Dense(units=90, activation='relu'),

    Dropout(0.2),

    Dense(units=30),

])

'''
# show model architecture

model.summary()
model.compile(optimizer='adam', loss='mse',

             metrics=['mae'])
# another method is to use LearningRateScheduler, reduce the learning rate by 10% every epoch

# annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

K = keras.callbacks

reduce_lr = K.ReduceLROnPlateau(monitor='val_accuracy', patience=7,

                                             verbose=1, factor=0.1, min_lr=0.00001)



early_stopping = K.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True,

                                 verbose=1, mode='auto')

#checkpointer = K.ModelCheckpoint(filepath = 'best_model.hdf5', monitor='val_mae',

                                 #verbose=1, save_weights_only=True)
# it is better to increase the batch size when the dataset is small

epochs = 600

batch_size = 512

history = model.fit(X_train, Y_train, validation_data=(x_val, y_val),

                   batch_size=batch_size, epochs=epochs, shuffle=True,

                   callbacks=[reduce_lr, early_stopping])
final_loss, final_mae = model.evaluate(x_val, y_val, verbose=0)

print("Final loss: {0:.4f}, final mae: {1:.4f}".format(final_loss * 48, final_mae * 48))
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['mae'], color='b', label="Training mae")

ax[1].plot(history.history['val_mae'], color='r',label="Validation mae")

legend = ax[1].legend(loc='best', shadow=True)
def plot_img_with_keypoints_after_training(nrows=4, ncols=4):

    selection = np.random.choice(len(X_test), size=(nrows*ncols), replace=False)

    images = X_test[selection]

    keypoints = output_pipe.inverse_transform(model.predict(images))

    fig, axes = plt.subplots(figsize=(nrows*2, ncols*2), nrows=nrows, ncols=ncols)

    for img, keypoint, ax in zip(images, keypoints, axes.ravel()):

        keypoint = keypoint.reshape(15,2)

        ax.imshow(img.reshape(96,96), cmap='gray')

        ax.scatter(keypoint[:,0], keypoint[:,1], marker='o', s=15)

        ax.axis('off')
plot_img_with_keypoints_after_training()
# save the weights to prevent training every time you open the kernel

model.save_weights("model.h5")
# after loading, you have to compile the model

#model.load_weights('/kaggle/input/facial-keypoints-detection2/model.h5')

#model.compile(optimizer='adam', loss='mean_squared_error',

 #            metrics=['mae'])



#final_loss, final_mae = model.evaluate(x_val, y_val, verbose=0)

#print("Final loss: {0:.4f}, final mae: {1:.4f}".format(final_loss * 48, final_mae * 48))
results = model.predict(X_test) 

results = output_pipe.inverse_transform(results)
print(results.shape)

print(type(results))
lookup_data = pd.read_csv("/kaggle/input/facial-keypoints-detection/IdLookupTable.csv")

row_ids = list(lookup_data['RowId'])

image_ids = list(lookup_data['ImageId'] - 1)

feature_names = list(lookup_data['FeatureName'])



feature_list = []

for feature in feature_names:

    feature_list.append(feature_names.index(feature))

    

predictions = []

for x,y in zip(image_ids, feature_list):

    predictions.append(results[x][y])

    

row_ids = pd.Series(row_ids, name = 'RowId')

locations = pd.Series(predictions, name = 'Location')

locations = locations.clip(0.0,96.0)

submission_result = pd.concat([row_ids,locations],axis = 1)

submission_result.to_csv('facial_keypoints.csv',index = False)

    
submission_result.shape