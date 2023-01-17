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
# Load training dataset 

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
# load testing dataset 

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
print(train_data.shape, test_data.shape)
# Data Exploration 
women = train_data.loc[train_data.Sex == 'female']['Survived']

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']['Survived']

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
from sklearn.ensemble import RandomForestClassifier



y = train_data['Survived']



feature = ['Pclass', 'Sex', 'SibSp', 'Parch'] # make a list for all the features included in the model 



# make a feature dataset for train and test dataset 

x = pd.get_dummies(train_data[feature])

x_test = pd.get_dummies(test_data[feature])

# building model

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)



model.fit(x, y)

predictions = model.predict(x_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index = False)
