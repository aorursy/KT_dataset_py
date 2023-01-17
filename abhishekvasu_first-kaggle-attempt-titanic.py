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
import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train["Sex"][train["Sex"]=='male'] = 0

train["Sex"][train["Sex"]=='female'] = 1



test["Sex"][test["Sex"]=='male'] = 0

test["Sex"][test["Sex"]=='female'] = 1



train["Age"][train["Age"].isnull()] = train["Age"].median()



train = train.drop(["PassengerId", "Name", "Ticket"], axis = 1)

test = test.drop(["PassengerId", "Name", "Ticket"], axis = 1)
test["Age"][test["Age"].isnull()] = test["Age"].median()
train = train.drop(["Cabin"], axis = 1)

test = test.drop(["Cabin"], axis = 1)
train["family_size"] = train["SibSp"] + train["Parch"] + 1

test["family_size"] = test["SibSp"] + test["SibSp"] + 1

test.head()
a = RandomForestClassifier(n_estimators = 10, max_depth = 10)

feature_set = train[["Pclass", "Sex", "Age", "family_size"]]

target = train["Survived"]

a.fit(feature_set, target)

print(a.score(feature_set, target))
feature_test = test[["Pclass", "Sex", "Age", "family_size"]]

prediction = a.predict(feature_test)