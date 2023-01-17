# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
trainData = pd.read_csv("../input/train.csv")
testData = pd.read_csv("../input/test.csv")
#convert things to categorical data
trainData.Sex = trainData.Sex.astype('category')
testData.Sex = testData.Sex.astype('category')
trainData['SexNum'] = trainData.Sex.cat.codes
testData['SexNum'] = testData.Sex.cat.codes
trainData.head()
trainData.describe()
trainData.describe(include=["O"])
trainData.Sex.value_counts().plot(kind="bar")
trainData.Pclass.value_counts().plot(kind="bar")
trainData.groupby(["Sex", "Survived"])["Survived"].count()
trainData[["PassengerId","Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "SexNum"]].corrwith(trainData.Survived)
from sklearn import tree
testFeatures = ["Pclass", "Age", "Fare", "SexNum"]
X = trainData[testFeatures].fillna(0).as_matrix()
Y = trainData["Survived"].fillna(0).as_matrix()
dtree = tree.DecisionTreeClassifier()
dtree.fit(X, Y)
testData["Survived"] = dtree.predict(testData[testFeatures].fillna(0).as_matrix())
predictions = testData[["PassengerId", "Survived"]]
predictions.to_csv("dtree_predictions.csv", index=False)
print(check_output(["ls", "."]).decode("utf8"))
