import pandas as pd

import numpy as np

from keras.utils import to_categorical
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
train.head(5)
train = train.replace("male", 0).replace("female", 1).replace("S", 0).replace("C", 1).replace("Q", 2)
train["Age"].fillna(train.Age.mean(), inplace=True)

train["Embarked"].fillna(train.Embarked.mode().iloc[0], inplace=True)
train["Ticket"] = 0

train["Name"] = 0

train["Cabin"] = 0
pid = train["PassengerId"]

train = train.apply(lambda x: (x-x.mean())/x.std(), axis=0).fillna(0)

train["PassengerId"] = pid

train.head(5)
train_data = train.values

xs = train_data[:, 2:]

y = train_data[:, 1]
y_binary = to_categorical(y)
test = test.replace("male", 0).replace("female", 1).replace("S", 0).replace("C", 1).replace("Q", 2)

test["Age"].fillna(test.Age.mean(), inplace=True)

test["Embarked"].fillna(test.Embarked.mode(), inplace=True)

test["Fare"].fillna(test.Fare.mean(), inplace=True)

test["Ticket"] = 0

test["Name"] = 0

test["Cabin"] = 0

test["Pclass"] = 0

pid = test["PassengerId"]

test = test.apply(lambda x: (x-x.mean())/x.std(), axis=0).fillna(0)

test["PassengerId"] = pid

test_data = test.values

xs_test = test_data[:, 1:]
from keras.models import Sequential

from keras.layers.core import Dense, Activation



model = Sequential([

        Dense(20, input_dim=10),

        Activation('sigmoid'),

        Dense(2),

        Activation('softmax')

        ])



model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(xs, y_binary, batch_size=5, verbose=1, epochs=20, validation_split=0.1)
predict = np.argmax(model.predict(xs_test), axis=1)
import csv

with open("predict_result_data.csv", "w") as f:

    writer = csv.writer(f, lineterminator='\n')

    writer.writerow(["PassengerId", "Survived"])

    for pid, survived in zip(test_data[:,0].astype(int), predict.astype(int)):

        writer.writerow([pid, survived])