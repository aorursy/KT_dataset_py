# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.svm import SVR

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



import matplotlib.pyplot as plt



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv("../input/data.csv")

df.head(3)
# Extract the training and test data

data = df.values

X = data[:, 1:]  # all rows, no label

y = data[:, 0]  # all rows, label only

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train[:3]
# Scale the data to be between -1 and 1

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

X_train[:3]
# Establish a model

model = SVR(C=1, cache_size=500, epsilon=1, kernel='rbf')

# Train the model - this will take a minute

model.fit(X_train, y_train)
# Score the model

model.score(X_test, y_test)
# Not a great score. Try other epsilons - this will take about 5 minutes.

epsilons = np.arange(1, 9)

scores = []

for e in epsilons:

    model.set_params(epsilon=e)

    model.fit(X_train, y_train)

    scores.append(model.score(X_test, y_test))

plt.plot(epsilons, scores)

plt.title("Epsilon effect")

plt.xlabel("epsilon")

plt.ylabel("score")

plt.show()
# Try other C's - This will take about a minute or so

model.set_params(epsilon=5)

Cs = [1e0, 1e1, 1e2, 1e3]

scores = []

for c in Cs:

    model.set_params(C=c)

    model.fit(X_train, y_train)

    scores.append(model.score(X_test, y_test))

plt.plot(Cs, scores)

plt.title("C effect")

plt.xlabel("C")

plt.ylabel("score")

plt.show()
# Best I can do with SVR appears to hover around 0.70, can we do better?