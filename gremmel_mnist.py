# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")
test_df = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")
train_data = train_df.get_values()
test_data = test_df.get_values()
train_imgs = train_data[:, 1:]
train_labels = train_data[:, 0]

test_imgs = test_data[:, 1:]
test_labels = test_data[:, 0]
train_imgs.shape, train_labels.shape
train_imgs = train_imgs.reshape((60000, 28, 28, 1))
test_imgs = test_imgs.reshape((10000, 28, 28, 1))
train_imgs.shape
import matplotlib.pyplot as plt

%matplotlib inline

plt.imshow(train_imgs[2,:, :,0])
from keras.utils import to_categorical
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)
from keras.models import *
from keras.layers import *
input_node = Input(shape=(28, 28, 1))

conv1 = Conv2D(32, (3, 3), strides=1, padding='same', activation='relu' )(input_node)
pool1 = MaxPooling2D((2,2), strides=(2,2))(conv1)

conv2 = Conv2D(64, (3, 3), strides=1, padding='same', activation='relu' )(pool1)
pool2 = MaxPooling2D((2,2), strides=(2,2))(conv2)

conv3 = Conv2D(128, (3, 3), strides=1, padding='same', activation='relu' )(pool2)
pool3 = MaxPooling2D((2,2), strides=(2,2), padding='same')(conv3)

flat = Flatten()(pool3)
out = Dense(10, activation='softmax')(flat)

model = Model(input_node, out)
model.summary()
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
from keras.callbacks import *

folder = 'logs/'
if not os.path.isdir(folder):
    os.makedirs(folder)


model_checkpoint = ModelCheckpoint(monitor='val_loss', 
                                   filepath=folder+'model{epoch:02d}.{loss:2.4f}.{acc:2.2f}.{val_loss:2.4f}.{val_acc:2.2f}.hdf5')

early_stop = EarlyStopping(monitor='val_loss', patience=3)

callbacks = [model_checkpoint, early_stop]

train_imgs = train_imgs / 255
test_imgs = test_imgs / 255
model.fit(train_imgs, train_labels, epochs=20, 
          validation_data=(test_imgs, test_labels), 
          callbacks=callbacks)
import cv2

test_img = cv2.imread('../input/vestidopreto/vestido-preto.jpg', 0)
plt.imshow(test_img)
test_img = cv2.resize(test_img, (28, 28))
plt.imshow(test_img)
test_img = test_img.reshape((28,28, 1))
test_img = np.expand_dims(test_img, 0)
test_img = test_img / 255
np.round(model.predict(test_img), 3)