import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import keras
df = pd.read_csv("../input/Churn_Modelling.csv")

df.sample(7)
X = df.iloc[:,3:13].values # we need these columns for ann algorithm.

y = df.iloc[:,13].values # this is just exited column
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X[:,1] = le.fit_transform(X[:,1])

X[:,1]
le = LabelEncoder()

X[:,2] = le.fit_transform(X[:,2])

X[:,2]
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features=[1])

X = ohe.fit_transform(X).toarray()

X = X[:,1:]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = .33, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)

X_test = sc.fit_transform(x_test)
from keras.models import Sequential

from keras.layers import Dense
model = Sequential()

model.add(Dense(7, init = "uniform", activation = "relu", input_dim = 11))

model.add(Dense(7, init = "uniform", activation = "relu"))

model.add(Dense(1, init = "uniform", activation = "sigmoid"))

model.compile(optimizer="adam", loss = "binary_crossentropy", metrics = ["accuracy"])

model.fit(X_train, y_train, epochs = 50)
y_pred = model.predict(X_test)

y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

print(cm)