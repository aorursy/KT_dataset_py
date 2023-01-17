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
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping
train_data = pd.read_csv('../input/digit-recognizer/train.csv')

test_data = pd.read_csv('../input/digit-recognizer/test.csv')
train_data.head()
X_train = train_data.drop(labels = ["label"],axis = 1) 

Y_train = train_data["label"]

Y_train = to_categorical(Y_train, num_classes = 10)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)
X_train.shape
X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2)
test_datagen = ImageDataGenerator(rescale = 1./255)
train = train_datagen.flow(X_train, Y_train, batch_size = 128)
test = test_datagen.flow(X_test, Y_test, batch_size = 128)
callback = EarlyStopping(monitor='loss', patience=8, restore_best_weights=True)
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 5, padding = 'same', activation = 'relu', input_shape = [28, 28, 1]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid'))
cnn.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, padding = 'same'))



cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding='valid'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding='valid'))

cnn.add(Droupout(0.5))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=256, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=10, activation='softmax'))
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model = cnn.fit_generator(train, epochs = 100, validation_data = test, callbacks = [callback])
cnn.evaluate(X_test,Y_test,verbose=2)
accuracy = model.history['accuracy']

loss = model.history['loss']

val_accuracy = model.history['val_accuracy']

val_loss = model.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs,accuracy,'r',label = 'training accuracy')

plt.plot(epochs,val_accuracy,'b',label = 'test accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
test_data.shape

test_data /= 256 

test_data = test_data.values.reshape(-1,28,28,1)

results = cnn.predict(test_data)
results = np.argmax(results,axis = 1)

results = pd.Series(results,name = 'Label')
submission = pd.concat([pd.Series(range(1,28001),name = 'ImageId'),results],axis = 1)
submission.to_csv('./submission.csv',index = False)