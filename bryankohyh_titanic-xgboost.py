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
train = pd.read_csv("/kaggle/input/titanic/train.csv")
feature_need = ["Sex","Age","SibSp","Parch","Fare","Embarked","Survived"]

need = ["Sex","Age","SibSp","Parch","Fare","Embarked"]

train = train[feature_need]



train["Age"] = train["Age"].fillna(train["Age"].mean())





X_train = train[need]

y_train = train["Survived"]
X_train["Embarked"] = X_train["Embarked"].fillna("S")
test = pd.read_csv("/kaggle/input/titanic/test.csv")



pasid = test["PassengerId"]
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()



encode = ["Sex","Embarked"]



for i in encode:

    X_train[i] = label_encoder.fit_transform(X_train[i])
feature_need = ["Sex","Age","SibSp","Parch","Ticket","Fare","Embarked"]

need = ["Sex","Age","SibSp","Parch","Fare","Embarked"]

test = test[feature_need]

test["Age"] = test["Age"].fillna(test["Age"].mean())

X_test = test[need]



for i in encode:

    X_test[i] = label_encoder.fit_transform(X_test[i])
from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(X_train, y_train)
pred = model.predict(X_test)



d = {"PassengerId": pasid, "Survived": pred}
prediction = pd.DataFrame(pasid, columns = ["PassengerId"])

prediction["Survived"] = pred
prediction.to_csv("/kaggle/working/submission.csv", index = False)