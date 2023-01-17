# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = "../input/digit-recognizer/train.csv"

test = "../input/digit-recognizer/test.csv"



train_data = pd.read_csv(train)

test_data = pd.read_csv(test).values



print(train_data.head())
from sklearn.model_selection import train_test_split





def data_split(data):

    y = data["label"]

    x = data.drop(columns=["label"]).values

    return (x, y)



def train_valid_split(x, y, SIZE=0.25, RANDOM_STATE=123):

    xTrain, xVal, yTrain, yVal = train_test_split(x, y, test_size=SIZE, random_state=RANDOM_STATE)

    return xTrain, xVal, yTrain, yVal



(x_train, y_train) = data_split(train_data)

# (xTest, yTest) = data_split(test_data)

xTrain, xVal, yTrain, yVal = train_valid_split(x_train, y_train)
# preprocessing X

xTrain_norm = xTrain

xVal_norm = xVal

xTest_norm = test_data
xTrain_norm = xTrain_norm.reshape(-1, 28, 28, 1)

xVal_norm = xVal_norm.reshape(-1, 28, 28, 1)

xTest_norm = xTest_norm.reshape(-1, 28, 28, 1)
# preprocessing Y

from keras.utils import to_categorical



yTrain_cat = to_categorical(yTrain.copy())

yVal_cat = to_categorical(yVal.copy())
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization



INPUT_SIZE = (28, 28, 1)

def model(inputSize = INPUT_SIZE, loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]):

    model = Sequential()

    model.add(Conv2D(filters=32, input_shape=inputSize, kernel_size=(2,2), padding="same", activation="relu"))

    model.add(Conv2D(filters=32, kernel_size=(2,2), padding="same", activation="relu"))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(filters=64, kernel_size=(2,2), padding="same", activation="relu"))

    model.add(Conv2D(filters=64, kernel_size=(2,2), padding="same", activation="relu"))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(filters=128, kernel_size=(2,2), padding="same", activation="relu"))

    model.add(Conv2D(filters=128, kernel_size=(2,2), padding="same", activation="relu"))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(10, activation="softmax"))

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model

    

model = model()
model.summary()
# image data generator



from tensorflow.keras.preprocessing.image import ImageDataGenerator



image_gen = ImageDataGenerator(rotation_range=20,

                               width_shift_range=0.1,

                               height_shift_range=0.1,

                               shear_range=0.1,

                               zoom_range=0.1,

                               horizontal_flip=False,

                               vertical_flip=False,

                              )

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



check_point = ModelCheckpoint("best_model.h5", monitor="val_accuracy", verbose=1, save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor="val_accuracy", patience=3, verbose=1, factor=0.5, min_lr=0.0001)
yVal_cat.shape
history = model.fit(image_gen.flow(xTrain_norm,yTrain_cat, batch_size=64),

                              epochs = 30, validation_data = (xVal_norm,yVal_cat),

                              verbose = 1,

                              callbacks=[check_point, reduce_lr])



# history = model.fit(image_gen.flow(xTrain_norm,yTrain_cat, batch_size=64),

#                               epochs = 30, validation_data = (xVal_norm,yVal_cat),

#                               verbose = 1,

#                               callbacks=[check_point, reduce_lr])
import matplotlib.pyplot as plt



fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

# ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

# ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
import itertools

from sklearn.metrics import confusion_matrix



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    

yPred = model.predict(xVal_norm) 

yPred_cat = np.argmax(yPred,axis = 1) 

yTrue = np.argmax(yVal_cat,axis = 1) 

confusion_mtx = confusion_matrix(yTrue, yPred_cat)

plot_confusion_matrix(confusion_mtx, classes = range(10))
results = model.predict(xTest_norm)

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)