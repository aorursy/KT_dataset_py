# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
train.head()
X = train.iloc[:,1:]

y = train['label']
plt.figure(figsize=(10,6))

sns.countplot(y)

plt.show()
from imblearn.over_sampling import SMOTE



sm = SMOTE(random_state=42)

X_res, y_res = sm.fit_resample(X, y)
plt.figure(figsize=(10,6))

sns.countplot(y_res)

plt.show()
Xm = X_res.reshape((X_res.shape[0],28,28,1))

ym = y_res
X_train, X_test, y_train, y_test = train_test_split(Xm,ym,test_size=0.1, random_state=42)
from keras.models import Sequential

from keras.layers import Conv2D, Dense, Flatten

from keras.utils import to_categorical
y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
model = Sequential()



model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(32, kernel_size=3, activation='relu'))

model.add(Flatten())

model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=10)
ym = to_categorical(ym)
model.fit(Xm, ym, epochs=3, batch_size=10)
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
X_test = np.array(test)
X_test.shape
X_test = X_test.reshape((X_test.shape[0],28,28,1))
y_test = model.predict(X_test)
y_test_2 = np.zeros(y_test.shape[0])
y_test[0]
for i in range(y_test.shape[0]):

    y_test_2[i] = np.argmax(y_test[i])
y_test_2
df = pd.DataFrame(y_test_2)
df.to_csv('samplesubmission.csv', header=False)