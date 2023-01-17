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
# Data Training

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

train_data.head()
# Data Testing

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_data.head()
# Calculate the percentage of female passengers in (train.csv)

women = train_data.loc[train_data.Sex == 'female']['Survived']

rate_women = sum(women)/len(women)

print("% of woman who survived", rate_women)
# Calculate the percentage of male passengers in (train.csv)

men = train_data.loc[train_data.Sex == 'male']['Survived']

rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
# Random Forest Model

from sklearn.ensemble import RandomForestClassifier



features = ['Pclass', 'Sex', 'SibSp', 'Parch']

y = train_data['Survived']

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({ 'PassengerId': test_data.PassengerId, 'Survived': predictions })

output.to_csv('my_ubmission.csv', index=False)

print("Submission was successfully saved!")