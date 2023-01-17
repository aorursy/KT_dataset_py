# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import sklearn

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score
adult = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?",

        skiprows=1)
adult.shape
adult.head()
adult["Country"].value_counts()
import matplotlib.pyplot as plt
adult["Age"].value_counts().plot(kind="bar")
adult["Sex"].value_counts().plot(kind="bar")
adult["Education"].value_counts().plot(kind="bar")
adult["Occupation"].value_counts().plot(kind="bar")
nadult = adult.dropna()
testAdult = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?",

        skiprows=1)
testSampleAdult = pd.read_csv("/kaggle/input/adult-pmr3508/sample_submission.csv",

        names=["Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?",

        skiprows=1)
testAdult = pd.concat([testAdult, testSampleAdult], axis=1, join='inner').sort_index()
nTestAdult = testAdult.dropna()
Xadult = nadult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
Yadult = nadult.Target
XtestAdult = nTestAdult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]
YtestAdult = nTestAdult.Target
Xadult.head()
Yadult.head()
XtestAdult.head()
YtestAdult.head()
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(Xadult,Yadult)
random_forest_scores = cross_val_score(random_forest, Xadult, Yadult, cv=10)
random_forest_mean = np.mean(random_forest_scores)
random_forest_mean
random_forest_YadultPred = random_forest.predict(Xadult)
random_forest_YadultPred
random_forest_accuracy = accuracy_score(Yadult,random_forest_YadultPred,normalize=True,sample_weight=None)
random_forest_accuracy
logistic_regression = LogisticRegression(solver='lbfgs')
logistic_regression.fit(Xadult,Yadult)
logistic_regression_scores = cross_val_score(logistic_regression, Xadult, Yadult, cv=10)
logistic_regression_mean = np.mean(logistic_regression_scores)
logistic_regression_mean
logistic_regression_YadultPred = logistic_regression.predict(Xadult)
logistic_regression_YadultPred
logistic_regression_accuracy = accuracy_score(Yadult,logistic_regression_YadultPred,normalize=True,sample_weight=None)
logistic_regression_accuracy
decision_tree = DecisionTreeClassifier()
decision_tree.fit(Xadult,Yadult)
decision_tree_scores = cross_val_score(decision_tree, Xadult, Yadult, cv=10)
decision_tree_mean = np.mean(decision_tree_scores)
decision_tree_mean
decision_tree_YadultPred = decision_tree.predict(Xadult)
decision_tree_YadultPred
decision_tree_accuracy = accuracy_score(Yadult,decision_tree_YadultPred,normalize=True,sample_weight=None)
decision_tree_accuracy