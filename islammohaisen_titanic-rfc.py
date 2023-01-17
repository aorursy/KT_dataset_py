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
# load data

train_df = pd.read_csv("../input/titanic/train.csv", index_col = 'PassengerId')

test_df = pd.read_csv("../input/titanic/test.csv")

index = test_df['PassengerId']     # for submission
train_df.describe()
test_df.describe()
train_df.head()
test_df.Fare.fillna(train_df.Fare.mean(), inplace=True)
train_df['Age'].fillna(value = train_df['Age'].mean(), inplace = True)

test_df['Age'].fillna(value = test_df['Age'].mean(), inplace = True)
train_df.Embarked.fillna(train_df.Embarked.mode()[0], inplace=True)
embarked_dummies = pd.get_dummies(train_df['Embarked'], prefix='Embarked')

train_df = pd.concat([train_df, embarked_dummies], axis=1)

train_df.drop('Embarked', axis=1, inplace=True)

# for test

embarked_dummies = pd.get_dummies(test_df['Embarked'], prefix='Embarked')

test_df = pd.concat([test_df, embarked_dummies], axis=1)

test_df.drop('Embarked', axis=1, inplace=True)
train_df.drop(columns = ['Ticket','Cabin'], inplace=True)

test_df.drop(columns = ['Ticket','Cabin'], inplace=True)
train_df['Sex'] = train_df['Sex'].map({'male':1, 'female':0})

test_df['Sex'] = test_df['Sex'].map({'male':1, 'female':0})
pclass_dummies = pd.get_dummies(train_df['Pclass'], prefix="Pclass")

train_df = pd.concat([train_df, pclass_dummies],axis=1)

train_df.drop('Pclass',axis=1,inplace=True)

# for test

pclass_dummies = pd.get_dummies(test_df['Pclass'], prefix="Pclass")

test_df = pd.concat([test_df, pclass_dummies],axis=1)

test_df.drop('Pclass',axis=1,inplace=True)
train_df['FamilySize'] = train_df['Parch'] + train_df['SibSp'] + 1

train_df['Singleton'] = train_df['FamilySize'].map(lambda s: 1 if s == 1 else 0)

train_df['SmallFamily'] = train_df['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)

train_df['LargeFamily'] = train_df['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

# for test

test_df['FamilySize'] = test_df['Parch'] + test_df['SibSp'] + 1

test_df['Singleton'] = test_df['FamilySize'].map(lambda s: 1 if s == 1 else 0)

test_df['SmallFamily'] = test_df['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)

test_df['LargeFamily'] = test_df['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
train_df.drop(columns = ['Name'], inplace=True)

test_df.drop(columns = ['Name'], inplace=True)
y = train_df['Survived']

x = train_df.drop(['Survived'], axis = 1)
# splitting

x_train, x_test, y_train, y_test = train_test_split(x,y , test_size=0.2, random_state=0)
final_model = RandomForestClassifier(n_estimators=100, max_leaf_nodes=105, max_depth = 6, random_state= 0)

final_model.fit(x_train, y_train)
train_df.columns
del test_df['PassengerId']
test_df.columns
predictions = final_model.predict(test_df)
sub_df = pd.DataFrame(data={

    'PassengerId': index,

    'Survived': predictions

})

sub_df.to_csv('submission.csv', index=False)