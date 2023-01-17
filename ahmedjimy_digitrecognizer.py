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



# TensorFlow ≥2.0 is required

import tensorflow as tf

from tensorflow import keras

assert tf.__version__ >= "2.0"



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten,Conv2D, MaxPooling2D, BatchNormalization



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



# Where to save the figures

PROJECT_ROOT_DIR = "."

CHAPTER_ID = "cnn"

IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)

os.makedirs(IMAGES_PATH, exist_ok=True)



def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):

    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)

    print("Saving figure", fig_id)

    if tight_layout:

        plt.tight_layout()

    plt.savefig(path, format=fig_extension, dpi=resolution)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#read train and test files in pandas DataFrame

train_df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_df = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
print(train_df.head())

print(train_df.shape)

print(test_df.shape)
Y_train_full = train_df['label'].astype('int32')

X_train_full = train_df.drop(labels = ['label'], axis=1).astype('float32')

X_test = test_df.astype('float32')
# delete for extra space

del train_df

del test_df



print(Y_train_full.shape)

print(X_train_full.shape)
# show the distribution of each class

print(Y_train_full.value_counts())

sns.countplot(Y_train_full)
# normalization, reshaping, and splitting 

X_train_full = X_train_full / 255.0

X_test = X_test / 255.0



X_train_full = X_train_full.values.reshape(X_train_full.shape[0], 28, 28, 1)

X_test = X_test.values.reshape(X_test.shape[0], 28, 28, 1)



X_train, x_val, Y_train, y_val = train_test_split(X_train_full, Y_train_full,

                                                  test_size=0.1, random_state=42)
print(X_train.shape)

print(x_val.shape)

print(Y_train.shape)

print(y_val.shape)

print(X_test.shape)
plt.imshow(X_train[0][:,:,0], cmap='gray')
# CNN model architecture (all these parameters might be tuned to achieve better results)

from functools import partial



DefaultConv2D = partial(Conv2D, kernel_size=3, activation='relu', padding='SAME')



model = Sequential([

    DefaultConv2D(filters=64, kernel_size=7, input_shape=[28,28,1]),

    BatchNormalization(),

    MaxPooling2D(pool_size=2),

    DefaultConv2D(filters=128),

    BatchNormalization(),

    DefaultConv2D(filters=128),

    BatchNormalization(),

    MaxPooling2D(pool_size=2),

    DefaultConv2D(filters=256),

    BatchNormalization(),

    DefaultConv2D(filters=256),

    BatchNormalization(),

    MaxPooling2D(pool_size=2),

    Flatten(),

    Dense(units=128, activation='relu'),

    Dropout(0.5),

    Dense(units=64, activation='relu'),

    Dropout(0.5),

    Dense(units=10, activation='softmax'),

])
# show model architecture

model.summary()
#optimizer = keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, amsgrad=False)

# sparse loss is used when there are multi-class 

# (note: use one-hot encoding for Y values if you didn't use sparse)

model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam',

              metrics=['accuracy'])             
K = keras.callbacks

reduce_lr = K.ReduceLROnPlateau(monitor='val_accuracy', patience=10,

                                             verbose=1, factor=0.1, min_lr=0.00001)

# another method is to use LearningRateScheduler, reduce the learning rate by 10% every epoch

# annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
# With data augmentation to prevent overfitting (accuracy 0.99286)



data_augmentation = keras.preprocessing.image.ImageDataGenerator(

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1)  # randomly shift images vertically (fraction of total height)



data_augmentation.fit(X_train)
epochs = 40

batch_size = 64

history = model.fit_generator(data_augmentation.flow(X_train,Y_train, batch_size=batch_size),

                              epochs=epochs, validation_data=(x_val,y_val),

                              steps_per_epoch=X_train.shape[0] // batch_size,

                              verbose=1, callbacks=[reduce_lr])
final_loss, final_acc = model.evaluate(x_val, y_val, verbose=0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
from sklearn.metrics import confusion_matrix



# Predict the values from the validation dataset

Y_pred = model.predict(x_val)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# compute the confusion matrix

Y_true = y_val

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

sns.heatmap(confusion_mtx, annot=True, fmt='d')
# predict results

results = model.predict(X_test)



# select the index with the maximum probability

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)