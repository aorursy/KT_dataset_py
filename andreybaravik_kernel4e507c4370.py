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

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test_data = pd.read_csv("../input/titanic/test.csv")

train_data = pd.read_csv("../input/titanic/train.csv")

women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)



men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
from sklearn.ensemble import RandomForestClassifier

import statistics



y = train_data["Survived"]



#features = ["Pclass", "Sex", "SibSp", "Parch"]

features = ["Pclass", "Sex", "SibSp", "Embarked", "Age", "Parch"]

#features = ["Pclass", "Sex", "SibSp", "Embarked", "Parch"]



i = 0;

for Z in train_data.Age:

    if np.isnan(Z):

        train_data.Age[i] = 999999

    i += 1;

        

i = 0;

for Z in test_data.Age:

    if np.isnan(Z):

        test_data.Age[i] = 999999

    i += 1;

    

i = 0;

for Z in train_data.Fare:

    if np.isnan(Z):

        train_data.Fare[i] = 999999

    i += 1;

        

i = 0;

for Z in test_data.Fare:

    if np.isnan(Z):

        test_data.Fare[i] = 999999

    i += 1;

    





X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=50000, max_depth=500, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission8.csv', index=False)

print("Your submission was successfully saved!")