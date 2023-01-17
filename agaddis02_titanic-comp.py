# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("../input/titanic/test.csv")

test_data.head()
women = train_data.loc[train_data.Sex == "female"]["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived: ", rate_women)
men = train_data.loc[train_data.Sex == "male"]["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived: ", rate_men)
mean_age = train_data['Age'].mean()

print("Mean age of: ", mean_age)
train_data.Age.fillna(value=mean_age, inplace=True)

test_data.Age.fillna(value=mean_age, inplace=True)
train_data['Sex_bool'] = train_data['Sex'].apply(lambda X: 1 if X == 'male' else 0)

train_data.head()
test_data['Sex_bool'] = test_data['Sex'].apply(lambda X: 1 if X == 'male' else 0)

test_data.head()
y = train_data["Survived"]



features = ["Pclass", "Sex_bool", "SibSp", "Parch", "Age"]

X = train_data[features]



X_test = test_data[features]
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(X, y)



my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

my_model.fit(X_train, y_train, 

             early_stopping_rounds=15, 

             eval_set=[(X_valid, y_valid)], 

             verbose=False)



predictions = my_model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived_pct': predictions})

output['Survived'] = output['Survived_pct'].apply(lambda x: 1 if x >= 0.90 else 0)

output_final = output.loc[:,['PassengerId','Survived']]

output_final.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")