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
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

train_data = train_data.append(test_data)



temp = train_data.iloc[891:]

temp.info()
import seaborn as sns



# women = train_data.loc[train_data.Sex == 'female', 'Survived']

# rate_women = sum(women)/len(women)



# print("% of women who survived:", rate_women)



sns.countplot(train_data['Sex'], hue=train_data['Survived'])

display(train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().round(3))
sns.countplot(train_data['Pclass'], hue=train_data['Survived'])

display(train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().round(3))
#sns.countplot(train_data['Fare'], hue=train_data['Survived'])



train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].median())



train_data[train_data['Fare'].isnull()]



train_data['Fare_new'] = pd.qcut(train_data['Fare'], 5)

train_data.head()
sns.countplot(train_data['SibSp'], hue=train_data['Survived'])

display(train_data[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().round(3))



train_data['Family'] = train_data['SibSp'] + train_data['Parch']

train_data.head()

display(train_data[['Family', 'Survived']].groupby(['Family'], as_index=False).mean().round(3))
sns.countplot(train_data['Embarked'], hue=train_data['Survived'])

display(train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().round(3))
#display(train_data[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().round(3))

train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())

train_data[train_data['Age'].isnull()]

train_data['Age_new'] = pd.qcut(train_data['Age'], 8)

train_data.head()
train_data['Embarked'] = train_data['Embarked'].fillna('C')

train_data[train_data['Embarked'].isnull()]

from sklearn.ensemble import RandomForestClassifier

from sklearn import tree



train_data_new = train_data.iloc[:891]

test_data_new = train_data.iloc[891:]



features  = ['Pclass', 'Sex', 'SibSp', 'Embarked','Fare_new', 'Family']



#X = train_data_new.drop(labels = ['PassengerId','Survived'], axis=1)

y = train_data_new['Survived']

X = pd.get_dummies(train_data_new[features])

#X_Submit = test_data.drop(labels=['PassengerId','Survived'],axis=1)

X_Submit = pd.get_dummies(test_data_new[features])

model = RandomForestClassifier(n_estimators=250, random_state=2, min_samples_split=20, oob_score=True)

model.fit(X, y)

print("score: %.5f" %(model.oob_score_))

predictions = model.predict(X_Submit)

predictions = predictions.astype(int)

#print(predictions)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

#print(output)

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")