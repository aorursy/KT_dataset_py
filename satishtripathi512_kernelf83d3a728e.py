# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/datathon19/train.csv')

df = pd.read_csv('../input/datathon19/tic_tac_toe_dataset.csv')

df_test = pd.read_csv('../input/datathon19/test.csv')



df.head()
df_train.head()
df_test.head()
df['class'].value_counts()
df['top_left_square'].value_counts()
df_train = df_train.drop(columns=['Id'])

df_test = df_test.drop(columns=['Id'])



df_train.head()
from sklearn.preprocessing import LabelEncoder

df = df.apply(LabelEncoder().fit_transform)



df.head()
X = df.iloc[:, 0:9].values

y = df.iloc[:, 9].values



print(X.shape)

print(y.shape)

print(df.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, min_samples_split = 10)

model.fit(X_train, y_train)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
import tensorflow as tf

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(3, input_dim=9, activation='relu', kernel_initializer='random_normal'))

model.add(Dense(9, activation='relu', kernel_initializer='random_normal'))

model.add(Dense(21, activation='relu', kernel_initializer='random_normal'))

model.add(Dense(9, activation='relu', kernel_initializer='random_normal'))

model.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



model.fit(X_train, y_train, epochs=100, batch_size=128)
from sklearn.preprocessing import LabelEncoder

df_test = df_test.apply(LabelEncoder().fit_transform)

df_test