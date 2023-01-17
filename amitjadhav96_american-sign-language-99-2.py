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
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ReduceLROnPlateau
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.preprocessing import image
from matplotlib.pyplot import imshow
from keras.applications.imagenet_utils import preprocess_input

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
train = pd.read_csv("../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv")
test = pd.read_csv("../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv")
train.head(10)
test.head(10)
x_train = train.drop(labels = ["label"],axis = 1) 
y_train = train["label"]
x_train.head()
y_train.head()
y_train.value_counts().plot.bar()
x_test = test.drop(labels = ["label"],axis = 1) 
y_test = test["label"]
x_train = train.drop(labels = ["label"],axis = 1) 
y_train = train["label"]
x_train.head()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train
x_train = x_train.values.reshape(-1,28,28,1)
x_test = x_test.values.reshape(-1,28,28,1)
plt.imshow(x_train[1].reshape(28,28))
dataAug = ImageDataGenerator(
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False)  # randomly flip images

dataAug.fit(x_train)
x_train.shape
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)
label_binarizer.classes_
class_names = ["A","B","C","D","E","F","G","H","I","K","L",'M','N','O','P','Q','R','S','T','U','V','W','X','Y']
y_train.shape
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state=42)
lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)
model = Sequential()

model.add(Conv2D(32, (3, 3), padding = "same", input_shape=(28,28,1), activation="relu"))
model.add(Conv2D(32, (3, 3), padding = "same", input_shape=(28,28,1), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides = (2,2)))


model.add(Conv2D(64, (3, 3),padding = "same", activation="relu"))
model.add(Conv2D(64, (3, 3),padding = "same", activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3),padding = "same", activation="relu"))
model.add(Conv2D(128, (3, 3),padding = "same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3),strides = (2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(24, activation="softmax"))
model.summary()
plot_model(model, to_file='Model.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))
model.compile(
    loss='categorical_crossentropy',
    optimizer="adam",
    metrics=['accuracy']
)
model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=30,
    validation_data=(x_val, y_val),
    shuffle=True,
    callbacks = [lr_reduce]
)

preds = model.evaluate(x_test,y_test)

print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

img_path = '../input/myhand/WIN_20200918_18_00_39_Pro (2).jpg'
img = image.load_img(img_path, target_size=(28, 28),  color_mode="grayscale")
imshow(img,cmap = "gray")
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
#x = x/255
x = np.vstack([x])
print("Predicted letter is: "+ class_names[np.argmax(model.predict(x))])
preds = model.predict(x_test)
n=5
plt.imshow(x_test[n].reshape(28,28),cmap="gray") 
plt.grid(False) 
print("Predicted letter is:",class_names[np.argmax(preds[n])],"\nActual Answer:",class_names[np.argmax(y_test[n])]) # Prediction - True Answer
