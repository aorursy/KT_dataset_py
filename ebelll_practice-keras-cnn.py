# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.
import tensorflow.keras.layers as Layers

import tensorflow.keras.activations as Activations

import tensorflow.keras.models as Models

import tensorflow.keras.optimizers as Optimizer

import tensorflow.keras.utils as Utils



import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from sklearn.metrics import confusion_matrix as CM
label_list = []

for label_name in os.listdir('../input/intel-image-classification/seg_train/seg_train'):

    label_list.append(label_name)

    

label_list
label_dict = {k:v for v, k in enumerate(label_list)}

label_dict
import cv2



def get_images(path):

    X = []

    y = []

    

    for label_folder in os.listdir(path):

        for image_file in os.listdir(path+r'/'+label_folder):

            image_path = path+r'/'+label_folder+r'/'+image_file

            image = cv2.imread(image_path)

            image = cv2.resize(image, (150, 150))

            X.append(image)

            y.append(label_dict[label_folder])

    return X, y

X_train, y_train = get_images('../input/intel-image-classification/seg_train/seg_train')

X_test, y_test = get_images('../input/intel-image-classification/seg_test/seg_test')
X_train, y_train = np.array(X_train), np.array(y_train)

X_test, y_test = np.array(X_test), np.array(y_test)

print('train dataset shape: ', X_train.shape, y_train.shape)

print('test dataset shape: ', X_test.shape, y_test.shape)
X_train, y_train = shuffle(X_train, y_train, random_state = 6969)
y_train
y_train_oh = Utils.to_categorical(y_train, dtype='int32')

y_train_oh
X_test, y_test = shuffle(X_test, y_test, random_state = 6969)

y_test_oh = Utils.to_categorical(y_test, dtype='int32')
f, ax = plt.subplots(5,5)

f.subplots_adjust(0, 0, 3, 3)

for i in range(0, 5):

    for j in range(0, 5):

        rnd_number = np.random.randint(0, len(X_train))

        ax[i, j].imshow(X_train[rnd_number])

        ax[i, j].set_title(label_list[y_train[rnd_number]])

        ax[i, j].axis('off')
import seaborn as sns

sns.countplot(y_train)
model = Models.Sequential()



model.add(Layers.Conv2D(200, kernel_size=(3,3), activation='relu', input_shape=(150, 150, 3)))

model.add(Layers.Conv2D(150, kernel_size=(3,3), activation='relu'))

model.add(Layers.MaxPool2D(5,5))

model.add(Layers.Conv2D(150, kernel_size=(3,3), activation='relu'))

model.add(Layers.Conv2D(120, kernel_size=(3,3), activation='relu'))

model.add(Layers.MaxPool2D(5,5))

model.add(Layers.Flatten())

model.add(Layers.Dense(200, activation='relu'))

model.add(Layers.Dense(150, activation='relu'))

model.add(Layers.Dense(80, activation='relu'))

model.add(Layers.Dropout(rate=0.5))

model.add(Layers.Dense(6, activation='softmax'))
model.summary()
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

import time

callback_list = [ModelCheckpoint(filepath='cnn_checkpoint.h5', monitor='val_loss', save_best_only=True),

                TensorBoard(log_dir="logs/{}".format(time.strftime('%A-%b-%d')))]
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train_oh, batch_size=64, epochs=30, validation_split=0.25,

                   callbacks = callback_list)
epochs = np.arange(1, 31)

plt.plot(epochs, history.history['accuracy'], label='train')

plt.plot(epochs, history.history['val_accuracy'], label='train')

plt.xlabel('epochs')

plt.ylabel('acc')

plt.legend()

plt.show
epochs = np.arange(1, 31)

plt.plot(epochs, history.history['loss'], label='train')

plt.plot(epochs, history.history['val_loss'], label='train')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend()

plt.show
model.load_weights('cnn_checkpoint.h5')
model.evaluate(X_test, y_test_oh)
print([label_list[i] for i in np.argmax(model.predict(X_test[:10]), axis=1)])

print([label_list[i] for i in y_test[:10]])
fig = plt.figure(figsize=(10,5))

fig.tight_layout()

for i in range(10):

    ax = fig.add_subplot(2, 5, i+1)

    plt.imshow(X_test[i])

    ax.set_title(label_list[y_test[i]])

    plt.axis('off')
pred = []

path = '../input/intel-image-classification/seg_pred/seg_pred'

for image_file in os.listdir(path):

    image_path = path+r'/'+image_file

    image = cv2.imread(image_path)

    image = cv2.resize(image, (150, 150))

    pred.append(image)
pred = np.array(pred)

pred_proba = np.argmax(model.predict(pred), axis=1)
pred.shape
fig = plt.figure(figsize=(20,15))

fig.tight_layout()

for i in range(25):

    ax = fig.add_subplot(5, 5, i+1)

    plt.imshow(pred[i])

    ax.set_title(label_list[pred_proba[i]])

    plt.axis('off')