# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns
train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')

train_data.head()
train_data.info()
test_data.info()
# Count plot of survived and not survived passengers

sns.countplot(x='Survived', data=train_data)
# Count plot of survived and not survived passengers by their sex

sns.countplot(x='Survived', hue= "Sex", data=train_data)
#  Count plot of survived and not survived passengers by their class

sns.countplot(x='Survived', hue= "Pclass", data=train_data)
# Check the age range

print(train_data["Age"].min(), train_data["Age"].max())
# Make bins for ages

bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]

train_data['AgeBin'] = pd.cut(train_data['Age'], bins)
# Plot survived passengers by age

train_data[train_data['Survived'] == 1 ]['AgeBin'].value_counts().sort_index().plot(kind='bar')
# Plot passengers who did not survive by age

train_data[train_data['Survived'] == 0 ]['AgeBin'].value_counts().sort_index().plot(kind='bar')
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]

columns=['Sex', 'SibSp', 'Parch', 'Pclass' ]

X = pd.get_dummies(train_data[columns])

X_test = pd.get_dummies(test_data[columns])



model = RandomForestClassifier(n_estimators=200, max_features=2, random_state=1)

model.fit(X, y)



predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")


