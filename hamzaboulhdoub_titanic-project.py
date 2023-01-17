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
train_set = pd.read_csv("/kaggle/input/titanic/train.csv")

train_set.head()
train_set.describe
test_set= pd.read_csv("/kaggle/input/titanic/test.csv")

test_set.head()
test_set.info
women = train_set.loc[train_set.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)
men = train_set.loc[train_set.Sex=='male']['Survived']

rate_men= sum(men)/ len(men)

print("% of men who survived :"+str(rate_men))
from sklearn.ensemble import RandomForestClassifier



y = train_set["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_set[features])

X_test = pd.get_dummies(test_set[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_set.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")