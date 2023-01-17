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
import keras

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from keras.layers import Input, Dense, Dropout, Flatten, add

from keras.layers import Conv2D, Activation, MaxPooling2D

from keras.layers import AveragePooling2D, BatchNormalization

from keras import backend as K

from keras.callbacks import ModelCheckpoint

import tensorflow as tf

from keras.models import Model

from keras.utils import plot_model

from keras.preprocessing. image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Input, Dense, PReLU, Dropout

from keras.models import Model

from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from keras.optimizers import SGD, Adam

import matplotlib.pyplot as plt

import cv2

from PIL import Image

import seaborn as sns

sns.set()

%matplotlib inline
%%time

train_data = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test_data = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
label_values=train_data['label'].value_counts().sort_index()

plt.figure(figsize=(5,3))

plt.title("Kanada MNIST label distributions")

sns.barplot(x=label_values.index, y=label_values)
img_train = train_data.drop(["label"], axis=1).values.reshape(-1, 28, 28, 1).astype('float32')

img_label = train_data["label"]

img_test = test_data.drop(["id"], axis=1).values.reshape(-1, 28, 28, 1).astype('float32')

print("img_train.shape = ", img_train.shape)

print("img_label.shape = ", img_label.shape)

print("img_test.shape = ", img_test.shape)
fig = plt.figure(figsize=(10, 10))

show_img = 0

for idx in range(img_train.shape[0]):

    plt.subplot(5, 5, show_img + 1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(img_train[idx].reshape(28, 28), cmap=plt.cm.binary)

    plt.title("label: %d" % img_label[idx])

    show_img += 1

    if show_img % 25 == 0:

        break
def build_model(input_shape=(28, 28, 1), num_classes = 10):

    input_layer = Input(shape=input_shape)

    x = Conv2D(32, (3,3), strides=1, padding="same", name="conv1")(input_layer)

    x = PReLU()(x)

    x = Conv2D(32, (3,3), strides=1, padding="same", name="conv2")(x)

    x = PReLU()(x)

    x = MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    x = Conv2D(64, (5,5), strides=1, padding="same", name="conv3")(x)

    x = PReLU()(x)

    x = MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

    x = Conv2D(64, (5,5), strides=1, padding="same", name="conv4")(x)

    x = PReLU()(x)

    x = MaxPooling2D(pool_size=(2, 2), name="pool4")(x)

    x = Conv2D(128, (5,5), strides=1, padding="same", name="conv5")(x)

    x = PReLU()(x)

    x = Conv2D(128, (5,5), strides=1, padding="same", name="conv6")(x)

    x = PReLU()(x)

    x = Flatten()(x)

    x = Dense(512, name="full1")(x)

    x = PReLU()(x)

    x = Dropout(0.5)(x)

    x = Dense(num_classes, activation='softmax', name="output")(x)

    model = Model(inputs=input_layer, outputs=x)

    return model
model = build_model()

model.summary()
from sklearn.model_selection import train_test_split

X_data = img_train / 255

Y_data = to_categorical(img_label)

x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.1)

print("x_train.shape = ", x_train.shape)

print("y_train.shape = ", y_train.shape)

print("x_test.shape = ", x_test.shape)

print("y_test.shape = ", y_test.shape)
train_datagen = ImageDataGenerator(

    rotation_range=9, 

    zoom_range=0.25, 

    width_shift_range=0.25, 

    height_shift_range=0.25

)

train_datagen.fit(x_train)

sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=1, factor=0.5, min_lr=0.00001)

checkpoint = ModelCheckpoint("bestmodel.model", monitor='val_accuracy', verbose=1, save_best_only=True)

earlyStopping = EarlyStopping(monitor='val_accuracy', patience=15, verbose=1, mode='min')
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
epochs = 100

batch_size = 64
history = model.fit_generator(

    train_datagen.flow(x_train, y_train, batch_size=batch_size),

    steps_per_epoch=x_train.shape[0] // batch_size,

    epochs=epochs,

    validation_data=(x_test, y_test),

    callbacks=[checkpoint, learning_rate_reduction])
plt.style.use("ggplot")

plt.figure(figsize=(16, 4))

plt.subplot(1, 2, 1)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.subplot(1, 2, 2)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
results=model.predict(img_test/255.0)

results=np.argmax(results, axis=1)
fig = plt.figure(figsize=(10, 10))

show_img = 0

for idx in range(img_test.shape[0]):

    plt.subplot(5, 5, show_img + 1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(img_test[idx].reshape(28, 28), cmap=plt.cm.binary)

    plt.title("predict: %d" % results[idx])

    show_img += 1

    if show_img % 25 == 0:

        break
sub=pd.DataFrame()

sub['id']=list(test_data.values[0:,0])

sub['label']=results

sub.to_csv("submission.csv", index=False)