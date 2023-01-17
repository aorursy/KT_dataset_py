import pandas as pd

import numpy as np

import itertools



from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, BatchNormalization, MaxPool2D

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras import optimizers

from tensorflow.keras.utils import to_categorical 

from tensorflow.keras.optimizers import RMSprop



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



np.random.seed()

sns.set(style='white', context='notebook', palette='deep')



print("Done")



import os

print(os.listdir("../input"))





train_data = pd.read_csv('../input/training111.csv')

train_data.head()

train_data.info()
train_data.describe()
label = train_data["Popularity"]





feature = train_data.drop(labels = ["Popularity"],axis = 1)



g = sns.countplot(label)


label = to_categorical(label, num_classes = 2)



feature_train, feature_val, label_train, label_val = train_test_split(feature, label, test_size = 0.3, random_state=42)



model_1 = Sequential()

model_1.add(Dense(40, activation = "sigmoid", input_shape = (2,)))

model_1.add(Dense(30, activation = "sigmoid"))

model_1.add(Dense(20, activation = "sigmoid"))

model_1.add(Dense(10, activation = "sigmoid"))

model_1.add(Dense(2, activation = "softmax"))





optimizer = optimizers.SGD(lr=0.08, clipnorm=1.)

model_1.compile(optimizer= optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])



print ("model defined")

history = model_1.fit(feature_train, label_train, batch_size = 5, epochs = 50, 

          validation_data = (feature_val, label_val), verbose = 1)
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)