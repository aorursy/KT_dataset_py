# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import numpy as np

import matplotlib

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

from mlxtend.plotting import plot_confusion_matrix







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Flatten, Dense
# Read the dataset

bank_df = pd.read_csv('/kaggle/input/predicting-churn-for-bank-customers/Churn_Modelling.csv')

bank_df.head()
X = bank_df.drop(labels=['CustomerId', 'Surname', 'RowNumber', 'Exited'], axis = 1)

y = bank_df['Exited']
X.head()
X.isna().any()
y.head()

label1 = LabelEncoder()

X['Geography'] = label1.fit_transform(X['Geography'])

label = LabelEncoder()

X['Gender'] = label.fit_transform(X['Gender'])

X.head()
X = pd.get_dummies(X, drop_first=True, columns=['Geography'])

X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
model = Sequential()

model.add(Dense(X.shape[1], activation='relu', input_dim = X.shape[1]))

model.add(Dense(128, activation='relu'))

model.add(Dense(1, activation = 'sigmoid'))

X.shape[1]

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train.to_numpy(), batch_size = 10, epochs = 10, verbose = 1)

y_pred = model.predict_classes(X_test)

y_pred

y_test

model.evaluate(X_test, y_test.to_numpy())

confusion_matrix(y_test, y_pred)

accuracy_score(y_test, y_pred)
