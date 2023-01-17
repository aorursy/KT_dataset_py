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
import seaborn as sns
import numpy as np
import pandas as pd
import os

print(os.listdir('/kaggle/input/digit-recognizer'))
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train.head()
x_train = train.drop('label', axis=1)
y_train = train['label'].astype('int32')
print('x_train shape: ', x_train.shape)
print('y_train shape: ', y_train.shape)
print('test shape: ', test.shape)
print('\nsamples in test: ', test.shape[0])
print('samples in train: ', x_train.shape[0])
print('Number of Examples Per Digit:\n', y_train.value_counts())
sns.countplot(y_train)
x_train /= 255.0
test /= 255.0
x_train = x_train.values.reshape(x_train.shape[0], 28, 28, 1)
test = test.values.reshape(test.shape[0], 28, 28, 1)
print('x_train shape: ', x_train.shape)
print('y_train shape: ', y_train.shape)
print('test shape: ', test.shape)
plt.figure(figsize=(12,10))
for img in range(10):
    plt.subplot(5, 5, img+1)
    plt.imshow(x_train[img].reshape((28, 28)), cmap='binary_r')
    plt.axis('off')
    plt.title('Label: ' + y_train[img].astype('str'))
plt.show()
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
y_train = to_categorical(y_train, 10)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.1, random_state=1)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=12, verbose=1,
          validation_data=(x_test, y_test))
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
train_stat = model.evaluate(x_train, y_train, verbose=0)
print('Train Loss:     ', round(train_stat[0], 5))
print('Train Accuracy: ', round(train_stat[1]*100, 4), '%')
print('Test Loss:      ', round(loss, 5))
print('Test Accuracy:  ', round(accuracy*100, 4), '%')
predictions = model.predict(x_test)
results = np.argmax(predictions, axis = 1)
results = pd.Series(results, name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submission.csv", index=False)