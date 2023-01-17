# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

def prepare_features(dataset):
    if "Survived" in dataset.columns:
        dataset = dataset.drop("Survived", axis=1)
    dataset = dataset.drop(["PassengerId", "Name", "Cabin", "Ticket"], axis=1)
    dataset["Sex"] = dataset["Sex"] == "Male"
    dataset["Embarked"] = 2*(dataset["Embarked"]=="Q") + (dataset["Embarked"]=="S")
    dataset["Age"] = dataset["Age"].fillna(dataset["Age"].mean())
    dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].mean())
    return dataset

train_features = prepare_features(train)
train_output = train["Survived"]

test_features = prepare_features(test)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(train_features, train_output)
output = test
output["Survived"] = clf.predict(test_features)
output = output[["PassengerId", "Survived"]]
output.to_csv("output.csv", index=False)
