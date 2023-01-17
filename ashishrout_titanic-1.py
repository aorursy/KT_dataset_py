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
train_data.isnull().sum()
test_data.isnull().sum()
from sklearn.impute import SimpleImputer



imp = SimpleImputer(missing_values=np.nan, strategy='mean')

train_data["Age"] = imp.fit_transform(train_data["Age"].values.reshape(-1,1))

test_data["Age"] = imp.fit_transform(test_data["Age"].values.reshape(-1,1))

test_data["Fare"] = imp.fit_transform(test_data["Fare"].values.reshape(-1,1))



imp_emb = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

train_data["Embarked"] = imp_emb.fit_transform(train_data["Embarked"].values.reshape(-1,1))





train_data.isnull().sum()

test_data.isnull().sum()

from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]

features = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]



X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])        # For categorical variables



model = RandomForestClassifier(n_estimators=500, max_depth=5,min_samples_leaf=2)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")

print(output)
