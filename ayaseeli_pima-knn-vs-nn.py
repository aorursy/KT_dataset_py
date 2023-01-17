import numpy as np 

import pandas as pd

from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from pandas import read_csv
pima = read_csv("../input/diabetes.csv")

pima = pima.reindex(np.random.permutation(pima.index))

X = pima[pima.columns[0:8]]

Y = pima[['Outcome']]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
i = 3

lscore = {}

while i < 20:

    model = KNeighborsClassifier(n_neighbors=i, weights='uniform')

    model.fit(X_train, y_train.values.ravel())

    score = model.score(X_train,y_train)

    lscore[i] = score

    i = i+2

lscore
model = KNeighborsClassifier(n_neighbors=3, weights='uniform')

model.fit(X_train, y_train.values.ravel())

y_pred = model.predict(X_test)

acc_test = metrics.accuracy_score(y_test,y_pred)

acc_test
from keras.models import Sequential

from keras.layers import Dense
model = Sequential()

model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))

model.add(Dense(8, init='uniform', activation='relu'))

model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model

model.fit(X.values, Y.values, nb_epoch=100, batch_size=10)

# evaluate the model
scores = model.evaluate(X_test.values, y_test.values)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))