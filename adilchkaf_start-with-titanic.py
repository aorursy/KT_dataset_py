# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

train_data.head()
train_data.info()
train_data.isnull().sum()
sns.heatmap(train_data.isnull(), yticklabels = False, cbar = False,cmap='coolwarm')
train_data['Age']=train_data['Age'].fillna(train_data['Age'].mean())
train_data['Age'] = train_data['Age'].astype(int)
test_data['Age']=test_data['Age'].fillna(test_data['Age'].mean())
test_data['Age'] = test_data['Age'].astype(int)
test_data['Fare']=test_data['Fare'].fillna(test_data['Fare'].mean())
sns.heatmap(train_data.isnull(), yticklabels = False, cbar = False,cmap='coolwarm')
test_data.head()
test_data.info()
train_data['Survived'].value_counts(normalize=True)


sns.countplot(train_data['Survived'])
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Fare", "Age", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=300, max_depth=30, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)

model.score(X,y)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")