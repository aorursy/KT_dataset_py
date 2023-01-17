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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']['Survived']

rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

x = pd.get_dummies(train_data[features])

x_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(x, y)

predictions = model.predict(x_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")

"""score: 0.77511"""
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

x = pd.get_dummies(train_data[features])

x_test = pd.get_dummies(test_data[features])



model = BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1))

model.fit(x, y)

predictions = model.predict(x_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")

"""score: 0.76555"""
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB



from sklearn.utils import shuffle



def train_and_test(model, train_x, train_y, features):

    model.fit(train_x, train_y)

    prediction = model.predict(pd.get_dummies(test_data[features]))

    accuracy = round(model.score(x, y)*100, 2)

    print("Accuracy: ", accuracy, "%")

    return prediction
features = ["Pclass", "Sex", "SibSp", "Parch"]

log_pred = train_and_test(LogisticRegression(), x, y, features)

svm_pred = train_and_test(SVC(), x, y, features)

knn_pred_4 = train_and_test(KNeighborsClassifier(n_neighbors = 4), x, y, features)

rf_pred = train_and_test(RandomForestClassifier(n_estimators = 100), x, y, features)

nb_pred = train_and_test(GaussianNB(), x, y, features)
train_data["Embarked"].value_counts(dropna=False)
train_data["Embarked"] = train_data["Embarked"].fillna("S")

train_data["Embarked"] = train_data["Embarked"].astype(str)

test_data["Embarked"] = test_data["Embarked"].fillna("S")

test_data["Embarked"] = test_data["Embarked"].astype(str)

train_data["Embarked"].value_counts(dropna=False)
train_data["Age"].fillna(train_data["Age"].mean(), inplace=True)

train_data["Age"] = train_data["Age"].astype(int)

train_data["AgeBand"] = pd.cut(train_data["Age"], 5)

test_data["Age"].fillna(test_data["Age"].mean(), inplace=True)

test_data["Age"] = test_data["Age"].astype(int)

test_data["AgeBand"] = pd.cut(test_data["Age"], 5)

print(train_data[["AgeBand","Survived"]].groupby(["AgeBand"], as_index=False).mean())
train_data.loc[train_data["Age"]<=16,'Age']=0

train_data.loc[(train_data["Age"]>16)&(train_data["Age"]<=32),'Age']=1

train_data.loc[(train_data["Age"]>32)&(train_data["Age"]<=48),'Age']=2

train_data.loc[(train_data["Age"]>48)&(train_data["Age"]<=64),'Age']=3

train_data.loc[train_data["Age"]>64,'Age']=4

train_data["Age"] = train_data["Age"].map({0:"Child",1:"Young",2:"Middle",3:"Prime",4:"Old"}).astype(str)



test_data.loc[test_data["Age"]<=16,'Age']=0

test_data.loc[(test_data["Age"]>16)&(test_data["Age"]<=32),'Age']=1

test_data.loc[(test_data["Age"]>32)&(test_data["Age"]<=48),'Age']=2

test_data.loc[(test_data["Age"]>48)&(test_data["Age"]<=64),'Age']=3

test_data.loc[test_data["Age"]>64,'Age']=4

test_data["Age"] = test_data["Age"].map({0:"Child",1:"Young",2:"Middle",3:"Prime",4:"Old"}).astype(str)
features = ["Pclass", "Sex", "SibSp", "Parch", "Embarked", "Age"]

x_v2 = pd.get_dummies(train_data[features])

x_v2.head()

test_x_v2 = pd.get_dummies(test_data[features])

test_x_v2.head()
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB



from sklearn.utils import shuffle



def train_and_test(model, train_x, train_y, test_x):

    model.fit(train_x, train_y)

    prediction = model.predict(test_x)

    accuracy = round(model.score(train_x, train_y)*100, 2)

    print("Accuracy: ", accuracy, "%")

    return prediction


log_pred_v2 = train_and_test(LogisticRegression(), x_v2, y, test_x_v2)

svm_pred_v2 = train_and_test(SVC(), x_v2, y, test_x_v2)

knn_pred_4_v2 = train_and_test(KNeighborsClassifier(n_neighbors = 4), x_v2, y, test_x_v2)

rf_pred_v2 = train_and_test(RandomForestClassifier(n_estimators = 100), x_v2, y, test_x_v2)

nb_pred_v2 = train_and_test(GaussianNB(), x_v2, y, test_x_v2)

output2 = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': rf_pred_v2})

output2.to_csv('my_submission2.csv', index=False)

print("Your submission was successfully saved!")
