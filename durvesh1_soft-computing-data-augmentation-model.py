import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/sign-language-mnist/sign_mnist_train.csv')

test = pd.read_csv('../input/sign-language-mnist/sign_mnist_test.csv')

train.head()
train.shape
from IPython.display import Image

Image("../input/sign-language-mnist/amer_sign2.png")
labels = train['label'].values

unique_val = np.array(labels)

np.unique(unique_val)
plt.figure(figsize = (18,8))

sns.countplot(x =labels)
from sklearn.preprocessing import LabelBinarizer

label_binrizer = LabelBinarizer()

labels = label_binrizer.fit_transform(labels)

labels
train.drop('label', axis = 1, inplace = True)

images = train.values
plt.style.use('grayscale')

fig, axs = plt.subplots(1, 5, figsize=(15, 4), sharey=True)

for i in range(5): 

        axs[i].imshow(images[i].reshape(28,28))

fig.suptitle('Grayscale images')
images =  images/255
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, stratify = labels, random_state = 7)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
import keras

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
num_classes = 24

batch_size = 125

epochs = 50
# CNN MODEL

model = Sequential()

model.add(Conv2D(64, kernel_size=(4,4), activation = 'relu', input_shape=(28, 28 ,1), padding='same' ))

model.add(Dropout(0.4))

model.add(MaxPooling2D(pool_size = (2, 2)))



model.add(Conv2D(64, kernel_size = (4, 4), activation = 'relu', padding='same' ))

model.add(Dropout(0.4))

model.add(MaxPooling2D(pool_size = (2, 2)))



model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))

model.add(Dropout(0.4))

model.add(MaxPooling2D(pool_size = (2, 2)))



model.add(Flatten())

model.add(Dense(128, activation = 'relu'))

model.add(Dense(num_classes, activation = 'softmax'))

model.compile(loss = keras.losses.categorical_crossentropy, optimizer='nadam',

              metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(shear_range = 0.25,

                                   zoom_range = 0.15,

                                   rotation_range = 15,

                                   brightness_range = [0.15, 1.15],

                                   width_shift_range = [-2,-1, 0, +1, +2],

                                   height_shift_range = [ -1, 0, +1],

                                   fill_mode = 'reflect')

test_datagen = ImageDataGenerator()
history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=batch_size)
plt.style.use('ggplot')

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.ylim(0.80, 1.05)

plt.title("Accuracy")

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.legend(['train','test'])

plt.show()
test_labels = test['label']

test.drop('label', axis = 1, inplace = True)

test_images = test.values/255

test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])

test_images = np.array([i.flatten() for i in test_images])

test_labels = label_binrizer.fit_transform(test_labels)

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

test_images.shape
# predictions

y_pred = model.predict(test_images)

from sklearn.metrics import accuracy_score

y_pred = y_pred.round()

accuracy_score(test_labels, y_pred)