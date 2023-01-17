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
# importing visualization libraries



%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt
# Loading the training data



train_df = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")
# Loading the testing data



test_df = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")
train_df.head()
test_df.head()
# dimensions of training data



train_df.shape
# dimension of testing data



test_df.shape
# basic stats about the training data



train_df.info()
# basic stats about the testing data



test_df.info()
# distribution of label



train_df['label'].value_counts()
train_df['label'].value_counts().plot(kind="bar")
# importing Keras libraries



import keras

from keras import utils

from keras import models

from keras import backend as K

from keras import optimizers

from keras.layers.core import (Dense, Dropout, Activation, Flatten, Reshape)

from keras.layers.convolutional import (Conv2D, MaxPooling2D)
# number of images per batch

BATCH_SIZE = 64



# number of epochs or no.f times the data has to be projected towards model

NP_EPOCHS = 50



# number of output classes

NP_CLASSES  = 10



# split the data into train and validation use 20% of data for validation

VALIDATION_SPLIT = 0.20



# optimizer has been passed to the model

OPTIMIZER = optimizers.RMSprop()
# dividing the training data into train and target



# training data

train = train_df.drop('label', axis=1)



# target data

target = train_df['label']



# test data

test = test_df.drop('id', axis=1)
train.head()
target.head()
test.head()
# Normalizing the train data



train /= 255
# Normalizing the test data



test /= 255
train.head()
test.head()
# reshaping the training data



train = train.values.reshape(-1, 28, 28, 1)
# taking a sample image from traing data



plt.imshow(train[42][:, :, 0])
# reshaping the test data



test = test.values.reshape(-1, 28, 28, 1)
# taking a sample image from test data



plt.imshow(test[21][:, :, 0])
# converting the target value into categorical values



target = utils.to_categorical(target, NP_CLASSES)
target[:5]
from keras.preprocessing import image

from keras.models import Sequential, Model

from keras.layers import Dense, Flatten, BatchNormalization, Activation, Dropout, LeakyReLU

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras import optimizers



# Output activation function

NP_OUTPUT_FUNC = "softmax"



# Neuron activation fuction

NP_DESCRIMINATOR_FUNC = "sigmoid"



# 25% Drouput or Dropping the signal or connection between the layers

NP_DROPOUT_FIRST = 0.25



# 20% Dropout or Dropping the signal or connetion between the layers

NP_DROPOUT_SECOND = 0.20
# Using the Keras 1 to construct the model (Not using the Functional API)



# Using the sequential model

model = Sequential()



# addint CNN Conv2D layer -  with a 32 5 X 5 filter

model.add(Conv2D(32, (5, 5), padding='same', input_shape=(28, 28, 1)))



# adding LeakyRelU as an activation function with 0.20 Leakage 

model.add(LeakyReLU(alpha=0.02))



# addint CNN Conv2D layer -  with a 32 5 X 5 filter

model.add(Conv2D(32, (5, 5)))



# adding LeakyRelU as an activation function with 0.20 Leakage 

model.add(LeakyReLU(alpha=0.02))



# adding Max pooling with 2 X 2 filters to generate the feature maps

model.add(MaxPooling2D(pool_size=(2, 2)))



# adding the first dropout as defined above - which generalize the model and avoid overfitting

model.add(Dropout(NP_DROPOUT_FIRST))



# addint CNN Conv2D layer -  with a 64 (No.f filters are increased) 3 X 3 filter

model.add(Conv2D(64, (3, 3), padding='same'))



# adding LeakyRelU as an activation function with 0.20 Leakage 

model.add(LeakyReLU(alpha=0.02))



# addint CNN Conv2D layer -  with a 64 3 X 3 filter

model.add(Conv2D(64, (3, 3)))



# adding LeakyRelU as an activation function with 0.20 Leakage

model.add(LeakyReLU(alpha=0.02))



# adding Max pooling with 2 X 2 filters to generate the feature maps

model.add(MaxPooling2D(pool_size=(2, 2)))



# adding the first dropout as defined above - which generalize the model and avoid overfitting

model.add(Dropout(NP_DROPOUT_FIRST))



# Flatten and passed to the dense network of our feature maps generated above

model.add(Flatten())

model.add(Dense(256))



# adding LeakyRelU as an activation function with 0.20 Leakage

model.add(LeakyReLU(alpha=0.02))



# adding the second dropout as defined above - which generalize the model and avoid overfitting

model.add(Dropout(NP_DROPOUT_SECOND))



model.add(Dense(256))

model.add(LeakyReLU(alpha=0.02))

model.add(Dropout(NP_DROPOUT_SECOND))



# providng the softmax activation function for output layers

model.add(Dense(NP_CLASSES))

model.add(Activation(NP_OUTPUT_FUNC))
# summary of the model



model.summary()
# plotting the model

from keras.utils import plot_model



plot_model(model, to_file='model.png', show_shapes=True)
# creating optimizer



# optimizer = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, decay=0.0)

optimizer
# compiling the model



model.compile(

    optimizer=optimizer,

    loss="categorical_crossentropy",

    metrics=["categorical_accuracy"]

)
# splitting the data for training and validation

from sklearn import model_selection



X_train, X_val, y_train, y_val = model_selection.train_test_split(train, target, test_size=0.20, random_state=123456789)
print("X training data shape: {}".format(X_train.shape))

print("Y training data shape: {}".format(y_train.shape))
print("X validation data shape: {}".format(X_val.shape))

print("Y validation data shape: {}".format(y_val.shape))
# creating a generator to get the data as small batches in a lazy format - Data augumentation



generator = image.ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=10,

    zoom_range=0.1,

    width_shift_range=0.1,

    height_shift_range=0.1,

    vertical_flip=False,

    horizontal_flip=False)
num_train_sequences = len(X_train)

num_val_sequences = len(X_val)



print("# training sequences: {}".format(num_train_sequences))

print("# validation sequences: {}".format(num_val_sequences))
# calculating number of training and validation steps per epoch

# for training

if (num_train_sequences % BATCH_SIZE) == 0:

    steps_per_epoch = int(num_train_sequences / BATCH_SIZE)

else:

    steps_per_epoch = int(num_train_sequences / BATCH_SIZE) + 1

    

# for validation    

if (num_val_sequences % BATCH_SIZE) == 0:

    validation_steps = int(num_val_sequences / BATCH_SIZE)

else:

    validation_steps = int(num_val_sequences / BATCH_SIZE) + 1    

    

print("# number of steps required for training: {}".format(steps_per_epoch))

print("# number of steps required for validation: {}".format(validation_steps))
# importing datatime package

import datetime



current_dt_time = datetime.datetime.now()

model_name = 'model_init' + '_' + str(current_dt_time).replace(' ', '').replace(':', '_') + '/'



if not os.path.exists(model_name):

    os.mkdir(model_name)
!ls -lrt
# call back: To save the best model based on validation categorical accuracy score



file_path = model_name + "model-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.h5"

checkpoint = ModelCheckpoint(filepath=file_path, monitor='val_categorical_accuracy', verbose=1, save_best_only=True,

                             save_weights_only=False, mode='auto', period=1)
# call back: ReduceOnPlateau - If a plateau found

 

LR = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=0.000001, verbose=1, cooldown=1)
# call back: Exponential Decay



def exponential_decay_fn(epoch):

    return 0.01 * 0.1**(epoch / 20)
def exponential_decay(lr, s):

    def exponential_decay_fn(epoch):

        return lr * 0.1**(epoch / s)

    return exponential_decay_fn



exponential_decay_fn = exponential_decay(lr=0.001, s=20)
lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
# adding all callbacks



callbacks = [

    # Model checkpoint

    checkpoint, 

    # ReduceOnPlateau

    LR, 

    # Learning Rate Scheduler

    lr_scheduler]
# fitting the model

VERBOSE = 1

history = model.fit_generator(generator.flow(X_train, y_train, batch_size=BATCH_SIZE), 

                             validation_data=generator.flow(X_val, y_val, batch_size=BATCH_SIZE),

                             epochs=NP_EPOCHS,

                             verbose=VERBOSE,

                             steps_per_epoch=steps_per_epoch,

                             validation_steps=validation_steps,

                             class_weight=None,

                             initial_epoch=0,

                             callbacks=callbacks)
# finding the best model



# best accuracy found

# best model: saving model to model_init_2019-07-0108_08_42.188814/model-00009-0.21558-0.93310-0.12834-0.96155.h5



values = {}

models = os.listdir(model_name)



for model in models:

    converted = model.replace(".h5", "")

    accuracy = float(converted.split("-")[-1])

    values.update({accuracy: model})

    

key = max(values, key = values.get)

best = values.get(key)



# Best model found among all the saved models

print("Best model found: {}".format(best))
# stored history information



history.history.keys()
plt.plot(history.history['categorical_accuracy'])

plt.plot(history.history['val_categorical_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
# summarize history for loss



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
# loading the above found model

from keras.models import load_model



model_path = model_name + best

print("Absolute path found: {}".format(model_path))
# loading the model



model = load_model(model_path)
# summary of the loaded model



model.summary()
# plotting the model done same as above



plot_model(model)
# predicting on test data



predictions = model.predict(test)
# sample predictions



predictions[:5]
# converting the above prediction into target classes



conversion = np.argmax(predictions, axis=1)
# converting into series



result = pd.Series(conversion, name="Label")
result.head()
# total length of result



print("No.f Id's available in the test dataset: {}".format(len(test_df['id'])))

print("No.f predictions done by the model on test data: {}".format(len(result)))
# creating a dataframe called submission to submit the data



submission = pd.concat([test_df['id'], result],axis = 1)
submission.head()
# creating submission file



submission.to_csv("submission.csv", index=False)
!ls -lrt