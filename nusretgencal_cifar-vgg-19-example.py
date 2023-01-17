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

        

import warnings

warnings.filterwarnings("ignore")

        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from keras.applications.vgg19 import VGG19

from keras.models import Sequential

from keras.utils import to_categorical

from keras.datasets import cifar10

from keras.optimizers import SGD

from keras.layers import Dense, Flatten

import matplotlib.pyplot as plt

import cv2

import numpy as np
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("x_train shape:", x_train.shape)

print("x_train sample:", x_train.shape[0])



numberofclass = 10



y_train = to_categorical(y_train, numberofclass)

y_test = to_categorical(y_test, numberofclass)



input_shape = x_train.shape[1::]
def resize_img(img):

    numberofimage = img.shape[0]

    new_array = np.zeros((numberofimage, 48,48,3)) # 3 rgb olduğunu belirtiyor

    for i in range(numberofimage):

        new_array[i] = cv2.resize(img[i,:,:,:], (48,48)) # 48x48 'e çıkarttık tüm imgleri

    return new_array
x_train = resize_img(x_train)

x_test = resize_img(x_test)

print("increased dim x_train", x_train.shape)
vgg = VGG19(include_top = False, weights = "imagenet", input_shape = (48,48,3)) # vgg19 modelimin weight'i imagenet datasetiyle eğitilmiş olsun, include_top ise fc layerimi almıyor false ile

vgg.summary()
vgg_layer_list = vgg.layers

model = Sequential()

for layer in vgg_layer_list:

    model.add(layer)

    

model.summary()
for layer in model.layers:

    layer.trainable = False

    

model.add(Flatten()),

model.add(Dense(128, activation= 'relu'))

model.add(Dense(numberofclass, activation ="softmax")) # output layer
model.summary()
x_train =x_train.astype(np.float32)/255.0

x_test= x_test.astype(np.float32)/255.0
opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])



hist = model.fit(x_train, y_train, 

                 validation_data=(x_test, y_test),

                 batch_size=35, 

                 epochs=10)

acc = max(hist.history['accuracy'])

val_acc = max(hist.history['val_accuracy'])



print ('Training Accuracy = ' + str(acc) )

print ('Validation Accuracy = ' + str(val_acc))



plt.plot(hist.history["loss"], label = "training_loss")

plt.plot(hist.history["val_loss"], label = "val_loss")

plt.legend()

plt.show()

plt.figure()

plt.plot(hist.history["accuracy"], label = "training_acc")

plt.plot(hist.history["val_accuracy"], label = "val_acc")

plt.legend()

plt.show()
from sklearn.metrics import classification_report, accuracy_score

predictions = model.predict(x_test, batch_size=256)

acc_score=accuracy_score(y_test.argmax(axis=1),predictions.argmax(axis=1))

print("Accuracy score:",acc_score)