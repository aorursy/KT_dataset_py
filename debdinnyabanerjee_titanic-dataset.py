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
train_data.info()
# there is missing data for the field Age, lets populate with the median age

age_median = train_data['Age'].median()

age_median
train_data['Age'].fillna(age_median,inplace=True)
# there is missing data also for Cabin and Embarked

train_data['Cabin'].value_counts()
train_data['Embarked'].value_counts()
# as majority of passengers embarked at S, lets populate missing data with S

train_data['Embarked'].fillna('S',inplace=True)

train_data['Embarked'].value_counts()
train_data.info()
# now we dont have any missing values except for Cabin

# lets import the test dataset

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
test_data.info()
# lets identify the most important factors affecting survival

features = ["Pclass","Sex","SibSp","Parch"]

X = pd.get_dummies(train_data[features])

y = train_data['Survived']

X_test = pd.get_dummies(test_data[features])
# import ML algorithm

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100,max_depth=5,random_state=1)
model.fit(X,y)
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId,'Survived':predictions})
output.to_csv('my_predictions.csv',index=False)
print('First project completed')