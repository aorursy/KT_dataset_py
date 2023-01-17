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
trainData.Sex = trainData.Sex.astype('category')
testData.Sex = testData.Sex.astype('category')
trainData['IsMale'] = trainData.Sex.cat.codes
testData['IsMale'] = testData.Sex.cat.codes
trainData.head(n=20)
display(trainData.describe().round(2))
display(trainData.describe(include=["O"]))
import matplotlib.pyplot as plt
trainData.Sex.value_counts(normalize=True).plot(kind="bar", title="Sex")
plt.show()
trainData.Pclass.value_counts().plot(kind="barh", title="Pclass")
plt.show()
trainData.Fare.plot(kind="density", title="Fare", xlim=(0,800))
plt.show()
trainData.groupby(["Sex", "Survived"])["Survived"].count()
trainData[["PassengerId","Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "IsMale"]].corrwith(trainData.Survived).round(4) * 100
testFeatures = ["IsMale", "Pclass", "Fare"]
X = trainData[testFeatures].fillna(0).as_matrix()
Y = trainData["Survived"].fillna(0).as_matrix()
from sklearn import tree
dtree = tree.DecisionTreeClassifier(max_depth=3)
dtree.fit(X, Y) #ALL OF THE MAGIC HAPPENS RIGHT HERE LADIES AND GENTLEMEN AND OTHERS
import graphviz
dot_data = tree.export_graphviz(dtree, out_file=None, feature_names=testFeatures)
graph = graphviz.Source(dot_data)
graph
testData["Survived"] = dtree.predict(testData[testFeatures].fillna(0).as_matrix())
predictions = testData[["PassengerId", "Survived"]]
predictions.to_csv("dtree_predictions.csv", index=False)
print(check_output(["ls", "."]).decode("utf8"))
