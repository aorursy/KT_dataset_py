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
train_data=pd.read_csv("/kaggle/input/titanic/train.csv")

train_data
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_data.head()
women = train_data[train_data.Sex == 'female']['Survived']

women
sum_women_survived = sum(women)

sum_women_survived
number_of_women = len(women)

number_of_women
rate_of_women_survival = sum_women_survived/number_of_women

rate_of_women_survival
train_data[train_data.Survived == 1].Sex.value_counts()/len(train_data[train_data.Survived == 1])
train_data[train_data.Survived == 1].Pclass.value_counts()/len(train_data[train_data.Survived == 1])
train_data[train_data.Survived ==1].Age.value_counts(bins=5).sort_index()/len(train_data[train_data.Survived==1])
female_count = train_data.Sex.value_counts()['female']

female_count
male_count = train_data.Sex.value_counts()['male']

male_count
train_data['male_female'] = [1 if x == 'female' else 0 for x in train_data.Sex]
train_data.corr()
from sklearn.ensemble import RandomForestClassifier
test_data.info()
test_data.columns
train_data.columns
train_data.info()
features = ['Pclass', 'Sex','SibSp','Parch']

features
y = train_data['Survived']
X=pd.get_dummies(train_data[features])

X
X_test = pd.get_dummies(test_data[features])

X_test
model = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state = 1)

model
model.fit(X,y)
predictions = model.predict(X_test)
predictions
len(predictions)
sum(predictions)/len(predictions)
output = pd.DataFrame({'PassengerId':test_data.PassengerId, 'Survived':predictions})

output
output.to_csv('my_submission.csv', index=False)