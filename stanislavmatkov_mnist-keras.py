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
%matplotlib inline



import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D

from keras.callbacks import ModelCheckpoint



print("Pandas", pd.__version__)

print("NumPy", np.__version__)
img_size = 28

batch_size = 64

validation_split = 0.1

epochs = 8
train_raw_dataset = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_raw_dataset = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
def preprocess(train_raw_dataset, test_raw_dataset):

    # Work with train data

    X_train = train_raw_dataset.drop(["label"], axis=1)

    X_train = X_train.values.astype("float32")

    X_train = X_train.reshape(X_train.shape[0], img_size, img_size, 1)

    X_train /= 255

    X_train = X_train.clip(0, 1)



    y_train = train_raw_dataset["label"]

    y_train = y_train.values

    y_train = np_utils.to_categorical(y_train, 10)

    

    # Split to train/validation set

    X_train, X_valid, y_train, y_valid = train_test_split(

        X_train, 

        y_train, 

        test_size=validation_split, 

        shuffle=True

    )

    

    # Work with test data

    X_test = test_raw_dataset.values.astype("float32")

    X_test = X_test.reshape(X_test.shape[0], img_size, img_size, 1)

    X_test /= 255

    X_test = X_test.clip(0, 1)

    

    return X_train, X_valid, y_train, y_valid, X_test



X_train, X_valid, y_train, y_valid, X_test = preprocess(train_raw_dataset, test_raw_dataset)
fig = plt.figure(figsize=(25, 4))

for i in range(20):

    ax = fig.add_subplot(2, 10, i+1)

    plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
datagen = ImageDataGenerator(

    rotation_range=10,

    width_shift_range=0.05,

    height_shift_range=0.05,

    zoom_range=0.05,

    data_format="channels_last",

    validation_split=validation_split

)



# compute quantities required for featurewise normalization

# (std, mean, and principal components if ZCA whitening is applied)

datagen.fit(X_train)



datagen_flow = datagen.flow(X_train, y_train, batch_size=batch_size)
fig = plt.figure(figsize=(25, 4))

for X_batch, y_batch in datagen_flow:

    for i in range(20):

        ax = fig.add_subplot(2, 10, i+1)

        plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))

    plt.show()

    break
input_shape = (img_size, img_size, 1)



model = Sequential()

model.add(Convolution2D(32, (5, 5), padding="same", input_shape=input_shape))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

model.add(Convolution2D(64, (5, 5), padding="same", input_shape=input_shape))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

model.add(Flatten())

model.add(Dense(1024))

model.add(Activation("relu"))

model.add(Dropout(0.5))

model.add(Dense(10))

model.add(Activation("softmax"))



model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])



print(model.summary())
train_history = model.fit_generator(

    datagen_flow, 

    epochs=epochs, 

    steps_per_epoch=(len(X_train) / batch_size), 

    validation_data=(X_valid, y_valid)

)
plt.plot(train_history.history["loss"], label="Training loss")

plt.plot(train_history.history["val_loss"], label="Validation loss")

plt.legend()

plt.ylabel('Loss')
plt.plot(train_history.history["loss"], label="Training accuracy")

plt.plot(train_history.history["val_loss"], label="Validation accuracy")

plt.legend()

plt.ylabel('Acc')
test = model.predict(X_test)

test_labels = np.argmax(test, axis=1)
fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):

    ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])

    plt.imshow(X_test[idx].reshape(28, 28))

    ax.set_label([test_labels[idx]])

    ax.set_title([test_labels[idx]])
results = pd.Series(test_labels, name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"), results],axis = 1)

submission.to_csv("mnist_prediction.csv", index=False)