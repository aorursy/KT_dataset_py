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
women = train_data.loc[train_data.Sex=='female']["Survived"]

percent_women = sum(women)/len(women)

print("Percent of women survived: ", percent_women)
men = train_data.loc[train_data.Sex=='male']["Survived"]

percent_men = sum(men)/len(men)

print("Percent of women survived: ", percent_men)
train_data["Fare"] = pd.to_numeric(train_data["Fare"], downcast = 'float')

train_data["Fare"]
test_data["Fare"] = pd.to_numeric(test_data["Fare"], downcast = 'float')

test_data["Fare"]
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch", "Fare"]



X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])
X
X_test
np.isnan(X_test)
np.where(np.isnan(X_test))
X_test = X_test.fillna(X_test.mean())
np.where(np.isnan(X_test))
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")