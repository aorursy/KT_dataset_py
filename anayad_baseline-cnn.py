import os

import numpy as np

import matplotlib.pyplot as plt

from keras.layers import *

from keras.models import Sequential

import pandas as pd

from keras.utils import np_utils

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# Dig_MNIST = pd.read_csv("../input/Kannada-MNIST/Dig-MNIST.csv")

sample_submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")

# test = pd.read_csv("../input/Kannada-MNIST/test.csv")

# train = pd.read_csv("../input/Kannada-MNIST/train.csv")

sample_submission.head()
train_df = pd.read_csv("../input/Kannada-MNIST/train.csv")

val_df = pd.read_csv("../input/Kannada-MNIST/Dig-MNIST.csv")

test_df = pd.read_csv("../input/Kannada-MNIST/test.csv")
test_df.head()
train_labels = train_df['label'].to_numpy()

train_df.drop('label', axis=1, inplace=True)

val_labels = val_df['label'].to_numpy()

val_df.drop('label', axis=1, inplace=True)
train_data = train_df.to_numpy().reshape((-1, 28, 28, 1))

val_data = val_df.to_numpy().reshape((-1, 28, 28, 1))
test_id = test_df['id']

test_df.drop('id', axis=1, inplace=True)

test_data = test_df.to_numpy()
test_data = test_data.reshape((-1, 28, 28, 1))
def plotImage(arr):

    fig = plt.figure(figsize=(8, 10))

    c=1

    for i in np.random.randint(0, arr.shape[0], 5):

        fig.add_subplot(1, 5, c)

        plt.imshow(arr[i, :, :, 0].astype(np.uint8), cmap='gray')

        c+=1

    plt.show()  
plotImage(train_data)
plotImage(test_data)
def baseline_model():

    model = Sequential()

    model.add(Conv2D(64, 3, 2, input_shape=(28, 28, 1)))

    model.add(BatchNormalization())

    model.add(Activation('elu'))



    model.add(MaxPool2D())

    model.add(Conv2D(128, 3, 2))

    model.add(BatchNormalization())

    model.add(Activation('elu'))



    model.add(Conv2D(256, 3, 2))

    model.add(BatchNormalization())

    model.add(Activation('elu'))

    model.add(MaxPool2D())

    model.add(Conv2D(512, 1, 2))

    model.add(BatchNormalization())

    model.add(Activation('elu'))



    model.add(Conv2D(1024, 1, 2))

    model.add(BatchNormalization())

    model.add(Activation('elu'))

    model.add(MaxPool2D())

    model.add(Flatten())

    model.add(Dense(2048, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(1024, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
model = baseline_model()

model.summary()
y_train = np_utils.to_categorical(train_labels, num_classes=10)

y_val = np_utils.to_categorical(val_labels, num_classes=10)
x_train = train_data/255.

x_val = val_data/255.

x_test = test_data/255.
#estop = EarlyStopping('val_loss', 0.001, 2)

lrdecay = ReduceLROnPlateau()
model.fit(x_train, y_train, batch_size=128, epochs=30, validation_data=(x_val, y_val), shuffle=True, callbacks=[lrdecay])
# Predict for test images

test_predictions = model.predict(x_test)
test_cls = np.argmax(test_predictions, axis=1)
sub_df = pd.DataFrame({"id":test_id, "label":test_cls})
sub_df.head()
sub_df.to_csv("submission.csv", index=False)