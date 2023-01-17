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


import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, CSVLogger

!pip install livelossplot

from livelossplot import PlotLossesKeras

TRAINING_LOGS_FILE = "training_logs.csv"

MODEL_SUMMARY_FILE = "model_summary.txt"

MODEL_FILE = "model.h5"

TRAINING_PLOT_FILE = "training.png"

VALIDATION_PLOT_FILE = "validation.png"

KAGGLE_SUBMISSION_FILE = "kaggle_submission.csv"



VERBOSITY = 1

EPOCHS = 100

BATCH_SIZE = 1024

CLASSES = 10

CHANNELS = 1

IMAGE_SIZE = 28

IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE

VALIDATION_RATIO = 0.1

# Load data

train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")



y = train["label"]

x = train.drop(labels = ["label"], axis = 1) 



x = x.values.reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)

test = test.values.reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)



y = to_categorical(y, num_classes=CLASSES)



x_training, x_validation, y_training, y_validation = train_test_split(x,

                                                                      y,

                                                                      test_size=VALIDATION_RATIO,

                                                                      shuffle=True)

# Data augmentation

data_generator = ImageDataGenerator(rescale=1./255,

                                    rotation_range=10,

                                    zoom_range=0.15, 

                                    width_shift_range=0.1,

                                    height_shift_range=0.1)

data_generator.fit(x_training)
model = Sequential()



model.add(Conv2D(filters=32,

                 kernel_size=(5,5),

                 padding='Same', 

                 activation='relu',

                 input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))

model.add(Conv2D(filters=32,

                 kernel_size=(5,5),

                 padding='Same', 

                 activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.5))



model.add(Conv2D(filters=64, kernel_size=(3,3),padding='Same', 

                 activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3,3),padding='Same', 

                 activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.5))



model.add(Flatten())

model.add(Dense(8192, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(2048, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(CLASSES, activation="softmax"))



model.compile(optimizer=RMSprop(lr=0.0001,

                                rho=0.9,

                                epsilon=1e-08,

                                decay=0.00001),

              loss="categorical_crossentropy",

              metrics=["accuracy"])
history = model.fit_generator(data_generator.flow(x_training,

                                                  y_training,

                                                  batch_size=BATCH_SIZE),

                              epochs=EPOCHS,

                              validation_data=(x_validation, y_validation),

                              verbose=VERBOSITY,

                              steps_per_epoch=x_training.shape[0] // BATCH_SIZE,

                              callbacks=[PlotLossesKeras(),

                                         CSVLogger(TRAINING_LOGS_FILE,

                                                   append=False,

                                                   separator=";")])

model.save_weights(MODEL_FILE)



# Testing

predictions = model.predict_classes(test, verbose=1)

pd.DataFrame({"ImageId":list(range(1,len(predictions)+1)),

              "Label":predictions}).to_csv(KAGGLE_SUBMISSION_FILE,

                                           index=False,

                                           header=True)



# Drawing plots

epochs = [i for i in range(1, len(history.history['loss'])+1)]



plt.plot(epochs, history.history['loss'], color='blue', label="training_loss")

plt.plot(epochs, history.history['val_loss'], color='red', label="validation_loss")

plt.legend(loc='best')

plt.title('training')

plt.xlabel('epoch')

plt.savefig(TRAINING_PLOT_FILE, bbox_inches='tight')

plt.close()
