# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_train
import matplotlib.pyplot as plt
split_data = []
for survived in [0,1]:
    split_data.append(df_train[df_train.Survived==survived])

temp = [i["Age"].dropna() for i in split_data]
plt.hist(temp, histtype="barstacked", bins=16)
df_train_tmp = df_train.ix[:, ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]]
df_train_tmp = df_train_tmp.replace("male", 0).replace("female", 1)
df_train_tmp = df_train_tmp.replace("C", 0).replace("Q", 1).replace("S", 2)
df_train_tmp["Age"].fillna(df_train_tmp["Age"].median(), inplace=True)
df_train_tmp["Fare"].fillna(df_train_tmp["Fare"].median(), inplace=True)
df_train_tmp["Embarked"].fillna(df_train_tmp["Embarked"].median(), inplace=True)
train = df_train_tmp.ix[:, ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
target = df_train_tmp.ix[:, "Survived"]

df_test_tmp = df_test.replace("male", 0).replace("female", 1)
df_test_tmp = df_test_tmp.replace("C", 0).replace("Q", 1).replace("S", 2)
df_test_tmp["Age"].fillna(df_test_tmp["Age"].median(), inplace=True)
df_test_tmp["Fare"].fillna(df_test_tmp["Fare"].median(), inplace=True)
df_test_tmp["Embarked"].fillna(df_test_tmp["Embarked"].median(), inplace=True)
test = df_test_tmp.ix[:, ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
import sklearn.tree
clf = sklearn.tree.DecisionTreeClassifier(max_depth=10)
clf = clf.fit(train, target)
predicted = clf.predict(test)
sklearn.tree.export_graphviz(clf, out_file="tree.dot", filled=True, rounded=True)
import csv
with open("predict_result_data.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(df_test["PassengerId"], predicted):
        writer.writerow([pid, survived])
