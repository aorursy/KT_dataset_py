# importing libraries



# linear algebra and data libraries

import pandas as pd

import numpy as np



# visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns



# scikit-learn data processing and evaluation libraries

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



# tensorflow & keras convolution neural network libraries

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, Lambda, MaxPooling2D

from tensorflow.keras.layers import Dense, Dropout, Flatten

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras import regularizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import to_categorical

from tensorflow.keras import backend as k

# listing input data files

import os

os.listdir('../input/digit-recognizer/')
# loading train and test data



train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')

print("train.shape: {}".format(train.shape))

print("test.shape: {}".format(test.shape))
X = train.drop(['label'], 1).values

y = train['label'].values



test_X = test.values
print("X.shape: {}".format(X.shape))

print("y.shape: {}".format(y.shape))



print("test_X.shape: {}".format(test_X.shape))
# normalization of pixels



X = X / 255.0

test_X = test_X / 255.0

np.max(X), np.max(test_X)
# reshaping data for convolution neural network



X = X.reshape(-1,28,28,1)

test_X = test_X.reshape(-1,28,28,1)
X.shape, test_X.shape
# encoding labels 



y = to_categorical(y)

print("y.shape: {}".format(y.shape))
# splitting train and test sets



X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                    test_size=0.1,

                                                    random_state=0)
print("X_train.shape: {}".format(X_train.shape))

print("y_train.shape: {}".format(y_train.shape))

print()

print("X_test.shape: {}".format(X_test.shape))

print("y_test.shape: {}".format(y_test.shape))
# looking data:



X_train_shaped = X_train.reshape(X_train.shape[0], 28, 28)



fig, axis = plt.subplots(1, 10, figsize=(28, 28))



for i, ax in enumerate(axis.flat):

    ax.imshow(X_train_shaped[i], cmap='binary')

    digit = y_train[i].argmax()

    ax.set(title = f"Number is: {digit}")
# hyperparameters:



batch_size = 16

epochs = 128

# Creating Model:



k.clear_session()



model = Sequential()



model.add(Conv2D(96, (2,2), strides=1, activation='relu', input_shape=(28, 28, 1)))

model.add(MaxPooling2D((2, 2), strides=1))

model.add(BatchNormalization(axis=1))



model.add(Conv2D(96, (2,2), strides=1, activation='relu'))

model.add(MaxPooling2D((3,3), strides=2))

model.add(BatchNormalization(axis=1))



model.add(Conv2D(96, (2,2), strides=1, activation='relu'))

model.add(Conv2D(96, (2,2), strides=1, activation='relu'))

model.add(Conv2D(96, (2,2), strides=1, activation='relu'))

model.add(BatchNormalization(axis=1))



model.add(MaxPooling2D((3,3), strides=2))



model.add(Flatten())



model.add(Dense(32, activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.1))

model.add(BatchNormalization())

model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.001)))

model.add(BatchNormalization())

model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.001)))

model.add(BatchNormalization())

model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.001)))

model.add(BatchNormalization())

model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.001)))

model.add(BatchNormalization())

model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.001)))

model.add(BatchNormalization())

model.add(Dense(10, activation='softmax'))



model.summary()
# creating artifical images:



image_generator = ImageDataGenerator(

                    featurewise_center=False,

                    samplewise_center=False,

                    featurewise_std_normalization=False,

                    samplewise_std_normalization=False,

                    zca_whitening=False,

                    rotation_range=15,

                    zoom_range=0.1,

                    width_shift_range=0.2,

                    height_shift_range=0.2,

                    horizontal_flip=False,

                    vertical_flip=False,

                    fill_mode="nearest")



image_generator.fit(X_train)

train_generated = image_generator.flow(X_train, y_train, batch_size=batch_size)

test_generated = image_generator.flow(X_test, y_test, batch_size=batch_size)

len(train_generated)
len(test_generated)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, batch_size=batch_size, validation_split=0.2, epochs=epochs)
history = model.fit(train_generated,

          epochs = 64, 

          steps_per_epoch = X_train.shape[0] // batch_size,

          validation_data = test_generated,

          validation_steps = X_test.shape[0] // batch_size)
from tensorflow.keras.utils import plot_model



plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
model.evaluate(X_test, y_test, verbose=0)
# displaying the accuracy and error for train and validation sets.



fig, ax = plt.subplots(2, 1, figsize=(20, 10))



ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label='validation loss', axes=ax[0])



legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label='Training accuracy')

ax[1].plot(history.history['val_accuracy'], color='r', label='Validation accuracy')



legend = ax[1].legend(loc='best', shadow=True)
fig = plt.figure(figsize=(10, 10))



pred = model.predict(X_test)



pred_label = np.argmax(pred, 1)

y_test_label = np.argmax(y_test, 1)



matrix = confusion_matrix(y_test_label, pred_label)



sns.heatmap(np.transpose(matrix), square=True, annot=True, cbar=False, cmap='Blues')
fig, axis = plt.subplots(4, 4, figsize=(10, 10))

plt.subplots_adjust(hspace = 0.6)



X_test_shaped = X_test.reshape(X_test.shape[0], 28, 28)



for i, ax in enumerate(axis.flat):

    ax.imshow(X_test_shaped[i], cmap='binary')

    ax.set(title = f"Number is {y_test[i].argmax()}\nPrediction is {pred[i].argmax()}")

pred_submit = model.predict_classes(test_X, verbose=1)
submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
submission['Label'] = pred_submit
submission.head()
submission.to_csv('Submission.csv', index=False)