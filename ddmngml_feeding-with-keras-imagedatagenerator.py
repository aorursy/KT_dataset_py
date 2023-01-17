import os
import math
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow
import seaborn as sns
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, ZeroPadding2D, BatchNormalization
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
from scipy.misc import imresize
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

warnings.simplefilter("ignore", category=DeprecationWarning)

%matplotlib inline
pd.set_option("display.max_rows", 10)
np.random.seed(42)
os.listdir("../input/")
dataset = pd.read_csv("../input/train.csv")
dataset.columns = ['filename', 'class'] # renaming to match ImageDataGenerator expectations
dataset.sample(5)
dataset.shape

batch_size = 128
subset = 500
target_size = (64, 64, 1) # set to grayscale

datagen = ImageDataGenerator(
    validation_split=.2,
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    horizontal_flip=False)

train_generator = datagen.flow_from_dataframe(
        dataframe=dataset.iloc[:subset],
        directory='../input/train',
        target_size=target_size[0:2],
        color_mode='grayscale', # this has to match the target_size parameter
        batch_size=batch_size,
        class_mode='categorical',
        interpolation='nearest')

num_classes = len(np.unique(train_generator.classes))

model = Sequential()
model.add(BatchNormalization(input_shape = target_size ))
model.add(Conv2D(filters=32, 
                 kernel_size=(7,7), 
                 activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(.25))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
opt = Adam(lr=0.02, decay=0.005)
model.compile(optimizer = opt , loss = "categorical_crossentropy", metrics=["categorical_accuracy"])
model.build()
model.summary()
epochs = 3

history = model.fit_generator(train_generator, epochs=epochs, steps_per_epoch=subset//epochs)
plt.plot(history.history['categorical_accuracy'])
plt.title('Model categorical_accuracy')
plt.ylabel('categorical_accuracy')
plt.xlabel('epoch')
plt.show()