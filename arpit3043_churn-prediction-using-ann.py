import os

import numpy as np

import seaborn as sns

import pandas as pd

import tensorflow as tf

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
tf.__version__
dataset = pd.read_csv('/kaggle/input/churn-predictions-personal/Churn_Predictions.csv')

X = dataset.iloc[:, 3:-1].values

y = dataset.iloc[:, -1].values

print(X)

print(y)
dataset.head()
dataset.tail()
dataset.describe()
dataset.corr()
for row in dataset.iterrows():

    print (row)
dataset.columns
sns.jointplot(x = 'RowNumber',y = 'CustomerId',data=dataset)
sns.jointplot(x='Tenure',y='Balance',data=dataset)
sns.jointplot(x='NumOfProducts',y='HasCrCard',data=dataset)
sns.jointplot(x='IsActiveMember',y='EstimatedSalary',data=dataset)
sns.jointplot(x='Age',y='Exited',data=dataset)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X[:, 2] = le.fit_transform(X[:, 2])

print(X)
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')

X = np.array(ct.fit_transform(X))

print(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
ann = tf.keras.models.Sequential()
print(ann)
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)
y_pred = ann.predict(X_test)

y_pred = (y_pred > 0.5)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)