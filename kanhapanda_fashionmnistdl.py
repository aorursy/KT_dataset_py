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
df = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
from sklearn.utils import shuffle

df = shuffle(df)

X = df.drop(['label'], axis = 1)

y = np.array(df['label'].astype(str).str.get_dummies())
X = np.reshape(np.array(X), (len(X),28,28,1))
img_rows, img_cols = 28, 28

num_classes = len(classes)
from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from keras.callbacks import EarlyStopping

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout



conv_model = Sequential()

conv_model.add(Conv2D(20, kernel_size=(3, 3),

                      activation='relu',

                      input_shape=(img_rows, img_cols, 1)))

conv_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))

conv_model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))



conv_model.add(Flatten())

conv_model.add(Dense(100, activation='relu'))

conv_model.add(Dense(num_classes, activation='softmax'))



conv_model.compile(loss='categorical_crossentropy',

                   optimizer = 'adam',

                   metrics=['accuracy'])

early_stopping_monitor = EarlyStopping(patience=3)

conv_model.fit(X,y,

                  batch_size = 20,

                  epochs = 4,

                 validation_split = 0.2,

              callbacks=[early_stopping_monitor])
preds = conv_model.predict(X)
def cal_accuracy(pred,y):

    c = total = 0

    for i in range(len(y)):

        total += 1

        if np.argmax(pred[i]) == np.argmax(y[i]):

            c += 1

    return c/total
cal_accuracy(preds,y)
test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')

test_x = test.drop(['label'], axis = 1)

test_x = np.reshape(np.array(test_x), (len(test_x),28,28,1))

test_y = np.array(test['label'].astype(str).str.get_dummies())
p = conv_model.predict(test_x)

cal_accuracy(p,test_y)
conv_model.evaluate(test_x,test_y)
dense_model = Sequential()

dense_model.add(Dense(1500, input_dim = 784, activation='relu'))

dense_model.add(Dense(1500, activation='relu'))

dense_model.add(Dense(1500, activation='relu'))



dense_model.add(Dense(10, activation='softmax'))
dense_model.compile(loss='categorical_crossentropy',

                   optimizer='adam',

                   metrics=['accuracy'])
X = X.reshape((len(X),28*28))
early_stopping_monitor = EarlyStopping(patience=3)

dense_model.fit(X,y,

                  batch_size = 100,

                  epochs = 4,

                  validation_split = 0.2,

                  callbacks=[early_stopping_monitor])
test_x = test_x.reshape((len(test_x), 28*28))

dense_model.evaluate(test_x,test_y)
pred = dense_model.predict(test_x)
pred[:5]
test_y[:5]
dense_model.summary()
from keras.utils import to_categorical

train_y_2 = to_categorical(df.label)
train_y_2[:5]