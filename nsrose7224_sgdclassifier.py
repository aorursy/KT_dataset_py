# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



import matplotlib.pyplot as plt



from subprocess import check_output

from datetime import time

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv("../input/dataset.csv")

print(df.shape)

df.head()
df = df.drop(["date", "time", "username"], axis=1)

df.head()
print(df.describe())

data = df.values

X = data[:, 1:]  # all rows, no label

y = data[:, 0]  # all rows, label only

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Scale the data to be between -1 and 1

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(loss="hinge", penalty="l2")
model.fit(X_train, y_train)
model.score(X_test, y_test)
n_iters = [5, 10, 20, 50, 100, 1000]

scores = []

for n_iter in n_iters:

    model = SGDClassifier(loss="hinge", penalty="l2", max_iter=n_iter)

    model.fit(X_train, y_train)

    scores.append(model.score(X_test, y_test))

  

plt.title("Effect of n_iter")

plt.xlabel("n_iter")

plt.ylabel("score")

plt.plot(n_iters, scores) 
# losses

losses = ["hinge", "log", "modified_huber", "perceptron", "squared_hinge"]

scores = []

for loss in losses:

    model = SGDClassifier(loss=loss, penalty="l2", max_iter=1000)

    model.fit(X_train, y_train)

    scores.append(model.score(X_test, y_test))

  

plt.title("Effect of loss")

plt.xlabel("loss")

plt.ylabel("score")

x = np.arange(len(losses))

plt.xticks(x, losses)

plt.plot(x, scores) 
np.random.shuffle(data)

from sklearn.model_selection import GridSearchCV



params = {

    "loss" : ["hinge", "log", "squared_hinge", "modified_huber"],

    "alpha" : [0.0001, 0.001, 0.01, 0.1],

    "penalty" : ["l2", "l1", "none"],

}



model = SGDClassifier(max_iter=1000)

clf = GridSearchCV(model, param_grid=params)



X = data[:, 1:]  # all rows, no label

y = data[:, 0]  # all rows, label only

clf.fit(X, y)

print(clf.best_score_)

print(clf.best_estimator_)