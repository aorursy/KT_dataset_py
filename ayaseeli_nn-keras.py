from keras.models import Sequential

from keras.layers import Dense

import numpy as np 

import pandas as pd



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from pandas import read_csv
cancer = read_csv("../input/data.csv")

del cancer["id"]

del cancer["Unnamed: 32"]
from sklearn.model_selection import  train_test_split

X = cancer[cancer.columns[1:31]]

d = {'M' : 0, 'B' : 1}

Y = cancer["diagnosis"].map(d).values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
model = Sequential()

model.add(Dense(34, input_dim=30, init='uniform', activation='relu'))

model.add(Dense(30, init='uniform', activation='relu'))

model.add(Dense(1, init='uniform', activation='sigmoid'))
from sklearn.preprocessing import StandardScaler

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

scaler = StandardScaler()

model.fit(scaler.fit_transform(X_train.values), y_train, nb_epoch=100, batch_size=10)
y_prediction = model.predict_classes(scaler.transform(X_test.values))

print ("\n\naccuracy" , np.sum(y_prediction == y_test) / float(len(y_test)))