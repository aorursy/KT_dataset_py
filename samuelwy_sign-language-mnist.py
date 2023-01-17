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
import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
train = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv')
test = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv')
train.head()
test.head()
# first column is the label column, separate it
train_labels = train['label']
test_labels = test['label']
train.drop('label', axis = 1, inplace = True)
test.drop('label', axis = 1, inplace = True)
train_labels.head()
train.head()
X = train.values
X_test = test.values
y = train_labels.values
y_test = test_labels.values
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size = 0.2, random_state = 42)
X_train.shape
X_valid.shape
fig, ax = plt.subplots(5,5, figsize = (15,10))

for i in range(5):
    for j in range(5):
        r = np.random.randint(len(X_train))
        ax[i,j].imshow(X_train[r].reshape(28, 28), cmap = 'gray')
        ax[i,j].axis('off')
        plt.tight_layout()    
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)
X_valid = X_valid.reshape(-1,28,28,1)

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1. / 255, 
                                   rotation_range = 40, 
                                   width_shift_range = 0.2, 
                                   height_shift_range = 0.2, 
                                   shear_range = 0.2, 
                                   zoom_range = 0.2, 
                                   fill_mode = 'nearest')

valid_datagen = ImageDataGenerator(rescale = 1. /255)
test_datagen = ImageDataGenerator(rescale = 1. /255)
from sklearn.preprocessing import LabelBinarizer
binarizer = LabelBinarizer()
y_train = binarizer.fit_transform(y_train)
y_valid = binarizer.fit_transform(y_valid)
y_test = binarizer.fit_transform(y_test)
model = keras.models.Sequential([
    keras.layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu', input_shape = (28,28,1)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(64, (3,3),padding = 'same', activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation = 'relu'),
    keras.layers.Dense(24, activation = 'softmax')
])

model.summary()

model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(train_datagen.flow(X_train,y_train, batch_size = 128), 
          epochs = 20, 
          steps_per_epoch = len(X_train) // 128, 
          validation_data = valid_datagen.flow(X_valid,y_valid, batch_size = 32),
          validation_steps = len(X_valid) // 32 , 
        callbacks = [keras.callbacks.EarlyStopping(patience = 2)])
## Plot the history of our model

# Get the different results
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(len(acc))

# Plot the training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title("Training and validation accuracy")
plt.figure()

# Plot the training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title("Training and validation loss")
plt.figure()
y_test.shape
# Evalulate on test data
model.evaluate(test_datagen.flow(X_test, y_test))
