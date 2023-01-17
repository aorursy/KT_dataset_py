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
from sklearn import datasets

import numpy as np

import pandas as pd

from sklearn import datasets
data =datasets.load_wine()
data.keys()
data['data'].shape
data['target']
pd.Series(data['target']).value_counts()
X = data['data']

#get dummy

Y = pd.get_dummies(data['target'])
X.shape
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X = scale.fit_transform(X)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y , test_size = 0.3)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers
model = tf.keras.Sequential()
#model

model.add(layers.Input(shape = x_train.shape[1]))

model.add(layers.Dense(169, activation='relu'))

model.add(layers.Dense(65, activation='relu'))

model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer='Adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
model.summary()
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 10, validation_split=0.2)
model.evaluate(x_test, y_test, batch_size=124)
pred = model.predict(x_test)
y_test_class = np.argmax(y_test.values,axis=1)

y_pred_class = np.argmax(pred,axis=1)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test_class, y_pred_class))
hist.history.keys()
b = pd.DataFrame(hist.history)
x = b.loss.argmin()

b.loc[x]
import matplotlib.pyplot as plt
plt.plot(b['loss'])

plt.plot(b['val_loss'])

plt.xlabel('epochs')

plt.ylim=([0, 1.0])

plt.legend(['loss','val_loss'])

plt.show()
plt.plot(b['accuracy'])

plt.plot(b['val_accuracy'])

plt.xlabel('epochs')

plt.ylim=([0, 1.0])

plt.legend(['accuracy','val_accuracy'])

plt.show()