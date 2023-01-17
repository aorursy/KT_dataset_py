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
train_data1 = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data1.head()
# load test data

test_data1 = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data1.head()
#find the pattern from traindata on basis of gender submission filef

female = train_data1.loc[train_data1.Sex == 'female']["Survived"]

rate_female = sum(female)/len(female)



print("% of female passengers who survived:", rate_female)
male = train_data1.loc[train_data1.Sex == 'male']["Survived"]

rate_male = sum(male)/len(male)



print("% of male passengers who survived:", rate_male)
from sklearn.ensemble import RandomForestClassifier



y = train_data1["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data1[features])

X_test = pd.get_dummies(test_data1[features])



model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=3)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data1.PassengerId, 'Survived': predictions})

output.to_csv('first_submission.csv', index=False)

print("First submission was successfully saved!")