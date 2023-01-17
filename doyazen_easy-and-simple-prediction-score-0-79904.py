

import pandas as pd

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanic/train.csv')

train.head()
train[['Sex','Survived' ]].groupby('Sex').count()
sns.countplot(x = 'Survived' ,data =train, hue = 'Sex')

plt.show()
plt.hist(train.Age)
sns.catplot(x= 'Sex' , y ='Age', data = train, kind= 'box', col = 'Survived')
train.info()
train.Age = train.Age.fillna(train.Age.mean())
train.head()
test = pd.read_csv('/kaggle/input/titanic/test.csv')

test.head()
test.info()
test.Age = test.Age.fillna(test.Age.mean())

test.Fare = test.Fare.fillna(test.Fare.mean())
test.info()
train.Sex = train.Sex.map({'male':1, 'female': 2})

train.head()
train.Embarked = train.Embarked.map({'C':1, 'S':2, 'Q':3})

train.head()
train.Embarked = train.Embarked.fillna(train.Embarked.mean())

train.Embarked = train.Embarked.astype('int')

train.head()
test.Sex = test.Sex.map({'male':1, 'female':2})

test.Embarked = test.Embarked.map({'C':1, 'S':2, 'Q':3})
test.head()
y = train.Survived
features = ['Pclass', 'Sex', 'Age','SibSp','Parch', 'Fare'] # 6 features
X = pd.get_dummies(train[features])
X.head()
X_test = pd.get_dummies(test[features])

X_test.head()
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X,y)
prediction = model.predict(X_test)

prediction
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': prediction})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")