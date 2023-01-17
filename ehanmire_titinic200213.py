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

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")

# PassengerId (sorted in any order)

# Survived (contains your binary predictions: 1 for survived, 0 for deceased)
# Ref: https://datascienceschool.net/view-notebook/16c28c8c192147bfb3d4059474209e0a/
# below code has been tested before 20200221
train.head()
num_cols = [col for col in train.columns if train[col].dtype in ['int64','float64']]

train[num_cols].describe()
cat_cols = [col for col in train.columns if train[col].dtype in ['O']]

train[cat_cols].describe()
for col in cat_cols[1:]:

    unq = np.unique(train[col].astype('str'))

    print('-'*50)

    print('column: {}, # of col: {}'.format(col, len(unq)))

    print('contents: {}'.format(unq))
# EDA idea

# plot (survived ~ pclass + age + sex)

# pclass is ordinal, age is continuous, survived & sex is categorical

# correlation ??  https://rfriend.tistory.com/405
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[train['Age'].isnull()]
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.tree import export_graphviz

from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris

iris_data = load_iris()

#iris_data.data; iris_data.target

X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target,

                                                   test_size = 0.2, random_state = 11)
# https://minjejeon.github.io/learningstock/2017/06/05/easy-one-hot-encoding.html



dt_clf = DecisionTreeClassifier(max_depth = 5,random_state = 156)

# dt_clf.fit(X_train, y_train)

#dt_clf.fit(train[['Pclass','SibSp','Parch','Sex']].to_numpy(), train['Survived'].to_numpy())

X_train = pd.concat([train[['Pclass','SibSp','Parch']], pd.get_dummies(train.Sex)], axis=1)



dt_clf.fit(X_train, train['Survived'])



X_train.columns
#export_graphviz(dt_clf, out_file='tree.dot', class_names=iris_data.target_names,

#                feature_names=iris_data.feature_names, impurity=True, filled=True)

export_graphviz(dt_clf, out_file = 'tree.dot', class_names = ['NotSurvived','Survived'],

                feature_names = X_train.columns, impurity=True, filled=True)



import graphviz

with open('tree.dot') as f:

    dot_graph = f.read()

graphviz.Source(dot_graph)