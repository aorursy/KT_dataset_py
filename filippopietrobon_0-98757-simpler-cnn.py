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

import seaborn as sb

from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



train_lab = train['label']

train.drop(columns =['label'], inplace = True)
plt.figure()

sb.countplot(train_lab)

plt.title('n° of images per class')





plt.figure()

plt.imshow(train.iloc[0].to_numpy().reshape(28,28))

plt.title('first sample')
train = train / 255

test = test / 255
X_train, X_test, y_train, y_test = train_test_split(train, train_lab, test_size = 0.2, random_state = 12)





X_train = X_train.values.reshape(-1,28,28,1)

y_train = tf.keras.utils.to_categorical(y_train)



X_test = X_test.values.reshape(-1,28,28,1)

y_test = tf.keras.utils.to_categorical(y_test)
model = Sequential()



model.add(Conv2D(32, kernel_size = (3,3), input_shape = (28,28,1)))

model.add(MaxPool2D())



model.add(Conv2D(64, kernel_size = (3,3)))

model.add(MaxPool2D())



model.add(Flatten())



model.add(Dense(128, activation  = 'relu'))

model.add(Dense(10, activation = 'softmax'))



model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(X_train, y_train,

          batch_size=128,

          epochs=20,

          validation_data = (X_test,y_test))
print('History keys: ',str(history.history.keys()))



plt.figure()

plt.plot(np.arange(len(history.history['loss'])), history.history['loss'])

plt.title('Loss Through Epochs')





plt.figure()

plt.plot(np.arange(len(history.history['loss'])), history.history['accuracy'])

plt.title('Acc Train')



plt.figure()

plt.plot(np.arange(len(history.history['loss'])), history.history['val_accuracy'])

plt.title('Acc Test')
test = test.values.reshape(-1,28,28,1)



y_test = model.predict(test)

y_test = np.argmax(y_test, axis = 1)
for i in range(0,10):

    plt.figure()

    plt.imshow(test[i][:,:,0])

    plt.title(str(y_test[i]))
results = pd.Series(y_test, name = 'Label')

IDs = pd.Series(range(1,28001), name = 'ImageId')



submission = pd.concat([IDs, results], axis = 1)



submission.head()



submission.to_csv('submission.csv', index = False)