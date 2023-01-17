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
trainData.head()
trainData.describe()
trainData.describe(include=["O"])
trainData.Sex.value_counts().plot(kind="bar")
trainData.Pclass.value_counts().plot(kind="bar")
trainData.groupby(["Sex", "Survived"])["Survived"].count()
trainData[["PassengerId","Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]].corrwith(trainData.Survived)
numericFeatures = ["Pclass", "Age", "Fare", "Parch", "SibSp"]

catFeatures = ["Sex", "Embarked"]

pd.get_dummies(trainData[catFeatures])
#select the features we're interested in

#categorical data needs to be handled differently

numericFeatures = ["Pclass", "Age", "Fare", "Parch", "SibSp"]

catFeatures = ["Sex", "Embarked"]



X_num_df = trainData[numericFeatures]

test_num_df = testData[numericFeatures]



#transform the categorical features with one-hot, then concatenate with numeric features

X_cat_df = pd.get_dummies(trainData[catFeatures])

X_df = pd.concat([X_num_df, X_cat_df], axis=1)

test_cat_df = pd.get_dummies(testData[catFeatures])

test_df = pd.concat([test_num_df, test_cat_df], axis=1)



X = X_df.fillna(0).as_matrix()

Y = trainData["Survived"].fillna(0).as_matrix()



X_df.head()
from sklearn import tree



dtree = tree.DecisionTreeClassifier(max_depth=4, criterion="entropy")

dtree.fit(X, Y)
import graphviz

dot_data = tree.export_graphviz(dtree, out_file=None, feature_names=X_df.columns)

graph = graphviz.Source(dot_data)

graph
testData["Survived"] = dtree.predict(test_df.fillna(0))
predictions = testData[["PassengerId", "Survived"]]

predictions.to_csv("dtree_predictions.csv", index=False)

print(check_output(["ls", "."]).decode("utf8"))