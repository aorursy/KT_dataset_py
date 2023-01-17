# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from cv2 import imread
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense, Flatten

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv')
test = pd.read_csv('../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 20))

ax1.imshow(imread('../input/sign-language-mnist/amer_sign2.png'))
ax1.axis('off')

ax2.imshow(imread('../input/sign-language-mnist/amer_sign3.png'))
ax2.axis('off')

ax3.imshow(imread('../input/sign-language-mnist/american_sign_language.PNG'))
ax3.axis('off')

plt.show()
train
test
X_train = train.drop('label', axis=1)
y_train = train['label']

X_test = test.drop('label', axis=1)
y_test = test['label']
X_train = np.array(X_train).reshape(27455, 28, 28, 1)
X_test = np.array(X_test).reshape(7172, 28, 28, 1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
X_train = X_train / 255.0
X_test = X_test / 255.0
idg = ImageDataGenerator()
train_gen = idg.flow(X_train, y_train, batch_size=64)
test_gen = idg.flow(X_test, y_test, batch_size=64)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(25, activation='softmax'))
model.compile(loss='categorical_crossentropy', metrics='categorical_accuracy', optimizer='adam')
history = model.fit_generator(train_gen, validation_data=test_gen, epochs=10)
results = history.history

for i in results:
    plt.plot(results[i])
    plt.title(i+' over epochs')
    plt.ylabel(i)
    plt.xlabel('epochs')
    plt.show()