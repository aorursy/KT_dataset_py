# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install split-folders

import splitfolders
import sys

from matplotlib import pyplot

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import Dropout

from keras.optimizers import SGD

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import RMSprop, Adam

from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
def summarize_diagnostics(history):

# plot loss

    plt.style.use("ggplot")

    plt.figure()

    N = epochs

    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")

    plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")

    plt.title("Loss")

    plt.xlabel("Epoch #")

    plt.ylabel("Loss")

    plt.legend(loc="upper left")

    # plot accuracy

    plt.style.use("ggplot")

    plt.figure()

    N = epochs

    plt.plot(np.arange(0, N), history.history["accuracy"], label="accuracy")

    plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_accuracy")

    plt.title("Accuracy")

    plt.xlabel("Epoch #")

    plt.ylabel("Accuracy")

    plt.legend(loc="upper left")

    # save plot to file
print("Class ok_front train count:",len(os.listdir('../input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data/train/ok_front')))

print("Class def_front train count:",len(os.listdir('../input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data/train/def_front')) )



print("Class def_front test count:",len(os.listdir('../input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data/test/def_front')))

print("Class ok_front test count:",len(os.listdir('../input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data/test/ok_front')))
splitfolders.ratio("../input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data/train", output="output", seed=1337, ratio=(.9, .1), group_prefix=None)
print("Class ok_front train count:",len(os.listdir('./output/train/ok_front')))

print("Class def_front train count:",len(os.listdir('./output/train/def_front')) )



print("Class ok_front train count:",len(os.listdir('./output/val/ok_front')))

print("Class def_front train count:",len(os.listdir('./output/val/def_front')) )
IMAGE_DIMS = (224, 224, 3)

train_data_dir = './output/train/'

validation_data_dir = './output/val/'

batch_size=64
train_datagen = ImageDataGenerator(rescale=1.0/255.0)

validation_datagen = ImageDataGenerator(rescale=1.0/255.0)



train_generator = train_datagen.flow_from_directory(

        train_data_dir,

        target_size=(IMAGE_DIMS[0], IMAGE_DIMS[1]),

        batch_size=batch_size,

        class_mode='categorical',

        shuffle=True)

validation_generator = validation_datagen.flow_from_directory(

        validation_data_dir,

        target_size=(IMAGE_DIMS[0], IMAGE_DIMS[1]),

        batch_size=batch_size,

        class_mode='categorical',

        shuffle=True)
nb_train_samples =5969 

nb_validation_samples = 664
# define cnn model

def define_model(h,w):

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',

    padding='same', input_shape=(h,w, 3)))

    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform',

    padding='same'))

    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform',

    padding='same'))

    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

    model.add(Dense(2, activation='softmax'))

    return model
model = define_model(IMAGE_DIMS[0],IMAGE_DIMS[1])

model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

epochs = 8

batch_size = 64

checkpoint = ModelCheckpoint("./weights.h5",monitor="loss",mode="min",save_best_only = True,verbose=1)

callbacks = [checkpoint]

history = model.fit_generator(train_generator,

    steps_per_epoch = nb_train_samples // batch_size,

    epochs = epochs,

    callbacks = callbacks,

    validation_data = validation_generator,

    validation_steps = nb_validation_samples // batch_size)
summarize_diagnostics(history)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

test_data_dir='../input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data/test/'

test_generator = test_datagen.flow_from_directory(

        test_data_dir,

        target_size=(IMAGE_DIMS[0], IMAGE_DIMS[1]),

        batch_size=batch_size,

        class_mode='categorical',

        shuffle=False)
model.load_weights("weights.h5")

class_labels = test_generator.class_indices

class_labels = {v: k for k, v in class_labels.items()}

classes = list(class_labels.values())

Y_pred = model.predict_generator(test_generator)

y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')

print(confusion_matrix(test_generator.classes, y_pred))

print('Classification Report')

target_names = list(class_labels.values())

print(classification_report(test_generator.classes, y_pred, target_names=target_names))



plt.figure(figsize=(8,8))

cnf_matrix = confusion_matrix(test_generator.classes, y_pred)



plt.imshow(cnf_matrix, interpolation='nearest')

plt.colorbar()

tick_marks = np.arange(len(classes))

_ = plt.xticks(tick_marks, classes, rotation=90)

_ = plt.yticks(tick_marks, classes)