# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.linear_model import SGDRegressor

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv("../input/data.csv")

# Extract the training and test data

data = df.values

X = data[:, 1:]  # all rows, no label

y = data[:, 0]  # all rows, label only

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Scale the data to be between -1 and 1

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
# Establish a model

model = SGDRegressor()
# Grid search - this will take about 1 minute.

param_grid = {

    'alpha': 10.0 ** -np.arange(1, 7),

    'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],

    'penalty': ['l2', 'l1', 'elasticnet'],

    'learning_rate': ['constant', 'optimal', 'invscaling'],

}

clf = GridSearchCV(model, param_grid)

clf.fit(X_train, y_train)

print("Best score: " + str(clf.best_score_))
