import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv("../input/deep-learning-az-ann/Churn_Modelling.csv")

df.head()
df.columns
df = df[['CreditScore','Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',

       'IsActiveMember', 'EstimatedSalary', 'Exited']]

df.head()
from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder() 

df['Gender']= le.fit_transform(df['Gender']) 

df.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = df.iloc[:, :-1].values

y = df.iloc[:, -1].values

X = scaler.fit_transform(X)

X
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from keras.models import Sequential

from keras.layers import Dense
model = Sequential()

model.add(Dense(20, activation='relu', input_shape=(9,)))

model.add(Dense(10, activation='relu', input_shape=(9,)))

model.add(Dense(5, activation='relu'))

model.add(Dense(2, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=100)
hist = model.history.history

plt.plot(hist['loss'])
model.evaluate(X_test, y_test)