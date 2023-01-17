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
# Python can be run as a script or by cell

# We will be mostly running python code in notebooks which runs code by cell

# Hashtag character allows us to comment code

# The hashtag character will tell the python interpreter ignore all characters on the same line to the right of it

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.model_selection import train_test_split

%matplotlib inline
train = pd.read_csv('../input/train.csv')
train.head()
train.count()
train.shape
type(train["Survived"])
train["Survived"].value_counts()
train["Pclass"].value_counts()
train["Parch"].value_counts()
firstclass = train["Pclass"] == 1
firstclass = train["Pclass"] == 1
train[firstclass]
fc_survived = train[firstclass]["Survived"]

fc_survived
fc_survived.value_counts()
secondclass = train["Pclass"] == 2

sc_survived = train[secondclass]["Survived"]

sc_survived.value_counts()
thirdclass = train["Pclass"] == 3

tc_survived = train[thirdclass]["Survived"]

tc_survived.value_counts()
fc_survived.value_counts().plot(kind = "bar")
fc_survived.value_counts().plot(kind = "bar")
our_cols = ["Parch" , "Pclass"]
X = train.loc[:, our_cols]
X
y = train.Survived
log_reg = LogisticRegression()
log_reg.fit(X, y)
test = pd.read_csv('../input/test.csv')
test.head()
X_new = test.loc[:, our_cols]
y_pred = log_reg.predict(X_new)
pd.DataFrame({"PassengerId": test.PassengerId, "Survived": y_pred})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train.shape
X_test.shape
log_reg2 = LogisticRegression()
log_reg2.fit(X_train, y_train)
y_pred2 = log_reg2.predict(X_test)
metrics.accuracy_score(y_test, y_pred2)