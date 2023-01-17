import cufflinks as cf

import numpy as np 

import pandas as pd 



RANDOM_STATE = 1234
local_path = "./data/"

kaggle_path = "/kaggle/input/killer-shrimp-invasion/"

example_submission_filename = "temperature_submission.csv"

train_filename = "train.csv"

test_filename = "test.csv"



base_path = kaggle_path



temperature_submission = pd.read_csv(base_path + example_submission_filename)

test = pd.read_csv(base_path + test_filename)

train = pd.read_csv(base_path + train_filename)
train.head()
test.head()
train_fill_na = train.fillna(method='ffill')

test_fill_na = test.fillna(method='ffill')
train_fill_na = train_fill_na[["pointid", "Presence"]]

test_fill_na = test_fill_na[["pointid"]]
train_fill_na.head()
test_fill_na.head()
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score
classifier = DecisionTreeClassifier(random_state=RANDOM_STATE)



X_full = train_fill_na[["pointid"]]

y_full = train_fill_na["Presence"]



n_splits = 10

scores = cross_val_score(classifier, X_full, y_full,

                                  scoring='roc_auc',

                                  cv=n_splits)



scores.mean()
classifier = DecisionTreeClassifier(random_state=RANDOM_STATE)

classifier.fit(X_full, y_full)
predictions = classifier.predict(test_fill_na)
temperature_submission['Presence'] = predictions

temperature_submission.to_csv('with_pointid.csv', index=False)
from sklearn import tree
tree.plot_tree(classifier)
train[train["Presence"] == 1].sort_values("pointid")["pointid"].values
temperature_submission[temperature_submission["Presence"] == 1].sort_values("pointid")["pointid"].values
temperature_submission[temperature_submission["pointid"] >= 2917768.5].sort_values("pointid")["pointid"].values