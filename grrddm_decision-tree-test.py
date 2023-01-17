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
cc = pd.read_csv("../input/creditcard.csv")

cc.head()
cc.Class.value_counts()
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split, cross_validate
train, test = train_test_split(cc, test_size=0.2)
tree = DecisionTreeClassifier()

cross_validate(tree, train.drop("Class", axis=1), train["Class"], cv=5, scoring=["precision", "recall"])
train["Class"].value_counts() / len(train)
tree = DecisionTreeClassifier()

train_split, cv_split = train_test_split(train, test_size=0.2)

tree.fit(train_split.drop("Class", axis=1), train_split["Class"])
from sklearn.metrics import precision_score, recall_score
neg_cv = cv_split.query("Class == 1")

precision_score(neg_cv["Class"], tree.predict(neg_cv.drop("Class", axis=1)))
tree.predict(neg_cv.drop("Class", axis=1))
precision_score(cv_split["Class"], tree.predict(cv_split.drop("Class", axis=1)))