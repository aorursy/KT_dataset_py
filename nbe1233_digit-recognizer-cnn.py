# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import keras

from keras.utils import to_categorical



### load csv

training_data_csv = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')



#### load labels

y = training_data_csv['label']

y = np.array(y, dtype=np.float32)

#print(y[0])

#print(y[3])

y_encoded = to_categorical(y)

#print(y_encoded[0])

#print(y_encoded[3])



### load training data

x = np.zeros((42000, 784), dtype=np.float32)



data = training_data_csv  

data = data.drop('label', 1) ### 1 = columns

for i in range(0, 42000):

    x[i] = data.iloc[i]

## reshape training data

x = np.reshape(x, (42000, 28 , 28))

## normalize data between 0 and 1

x = x/255.0



#import matplotlib.pyplot as plt

#print(y[0], y[35], y[41999])

#plt.imshow(x[0])

#plt.imshow(x[35])

#plt.imshow(x[41999])

#print(x.shape)
###load test data

test_data_csv = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

###print(len(test_data_csv['pixel0'])) ###28000

x_test = np.zeros((28000, 784), dtype=np.float32)

for i in range(0, 28000):

    x_test[i] = test_data_csv.iloc[i]

### normalize to between 0 and 1

x_test = np.reshape(x_test, (28000, 28, 28))

x_test = x_test / 255.0

#plt.imshow(x_test[39])

#print(y_encoded.shape) ### 42000, 10
from keras import Sequential

from keras.layers import Dense, Conv2D, Reshape, Flatten, MaxPool2D, Dropout



model = Sequential()

model.add(Reshape((28,28,1), input_shape=(28,28)))

model.add(Conv2D(32, (3,3), activation='relu', padding='same'))

model.add(Dropout(.2))

model.add(MaxPool2D())



model.add(Conv2D(64, (3,3), activation='relu', padding='same'))

model.add(Dropout(.2))

model.add(MaxPool2D())



model.add(Conv2D(128, (3,3), activation='relu', padding='same'))

model.add(Dropout(.2))

model.add(MaxPool2D())



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dense(256, activation='relu'))

model.add(Dense(10, activation='sigmoid'))

print(model.summary())
my_callbacks=[keras.callbacks.EarlyStopping(patience=2),]

model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy']

)

model.fit(x, y_encoded, epochs=40, validation_split=0.2, callbacks=my_callbacks)
y_test = model.predict(x_test)
y_test_decoded = np.zeros((28000,), dtype=np.int8)

for i in range(0, 28000):

    y_test_decoded[i] = np.argmax(y_test[i])
plt.imshow(x_test[1])
a = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

a['Label'] = y_test_decoded

a.to_csv('/kaggle/working/submit.csv',sep=',', index=False)
a = pd.read_csv('/kaggle/working/submit.csv')

print(a)