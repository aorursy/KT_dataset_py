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
dtypes = {'PassengerId': object, 

          'Survived': np.int, 

          'Pclass': np.float, 

          'Name': object, 

          'Sex': object, 

          'Age': np.float, 

          'SibSp': np.float,

          'Parch':np.float, 

          'Ticket': object, 

          'Fare': np.float, 

          'Cabin': object, 

          'Embarked': object}

train = pd.read_csv("../input/train.csv", dtype=dtypes)

test = pd.read_csv("../input/test.csv", dtype=dtypes)
train["Age"] = train["Age"].fillna(train.Age.mean())

train = train[train.Embarked.notnull()]
import sklearn.ensemble
rfr = sklearn.ensemble.RandomForestClassifier(n_estimators=300, n_jobs = -1)
rfr.fit(pd.get_dummies(train.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)), 

                       train.loc[:,["Survived"]])
test["Age"] = test["Age"].fillna(test.Age.mean())

test["Fare"] = test["Fare"].fillna(test.Fare.mean())
Survived = rfr.predict(pd.get_dummies(test.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)))
out = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": Survived.astype(int)})
out.to_csv("TitanicOut.csv", index=False)