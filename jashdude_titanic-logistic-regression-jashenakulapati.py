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

test = pd.read_csv("../input/titanic-machine-learning-from-disaster/test.csv")

train = pd.read_csv("../input/titanic-machine-learning-from-disaster/train.csv")
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train.head()

test.head()
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap="viridis")
sns.heatmap(test.isnull(), yticklabels=False, cbar=False, cmap="viridis")
sns.set_style("whitegrid")
sns.countplot(x="Survived", hue="Pclass", data=train)
train["Age"].plot.hist()
test["Age"].plot.hist()
sns.countplot(x="SibSp", data=train)
train["Fare"].plot.hist(bins=40, figsize=(10,4))
sns.boxplot(x="Pclass", y="Age", data=train)
sns.boxplot(x="Pclass", y="Age", data=test)
train.groupby("Pclass")["Age"].mean()
test.groupby("Pclass")["Age"].mean()
def impute_train_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass==1:

            return 38

        elif Pclass ==2:

            return 29

        else:

            return 25

    else:

        return Age



def impute_test_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass==1:

            return 40

        elif Pclass ==2:

            return 28

        else:

            return 24

    else:

        return Age
test["Age"]=test[["Age", "Pclass"]].apply(impute_test_age, axis=1)
train["Age"]=train[["Age", "Pclass"]].apply(impute_train_age, axis=1)
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap="viridis")
sns.heatmap(test.isnull(), yticklabels=False, cbar=False, cmap="viridis")
train.drop("Cabin", axis=1, inplace=True)
test.drop("Cabin", axis=1, inplace=True)
sex=pd.get_dummies(train["Sex"], drop_first=True)
sex_test=pd.get_dummies(test["Sex"], drop_first=True)
embark = pd.get_dummies(train["Embarked"], drop_first=True)
embark_test = pd.get_dummies(test["Embarked"], drop_first=True)
train=pd.concat([train, sex, embark], axis=1)

train.head()
test=pd.concat([test, sex_test, embark_test], axis=1)

test.head()
train.drop(["Sex","Embarked", "Name", "Ticket"], axis=1, inplace=True)

train.head()
test.drop(["Sex","Embarked", "Name", "Ticket"], axis=1, inplace=True)

test.head()
X_train = train.drop("Survived", axis=1)

Y_train = train["Survived"]

X_train.head()
Y_train.head()
test.head()
X_test = test.fillna(0)

X_test.head()
X_test[X_test["Fare"].isnull()==True]
#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
predictions=logreg.predict(X_test)
gender_submission = test[["PassengerId"]]

gender_submission.head()
gender_submission["Survived"] = predictions

gender_submission.head()

gender_submission.to_csv("gender_submission.csv", index=False)
#from sklearn.metrics import classification_report
#print(classification_report(y_test, predictions))
#from sklearn.metrics import confusion_matrix
#confusion_matrix(y_test, predictions)