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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
train_data.info()
train_data.describe()
sns.heatmap(train_data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
# There are some missing values in Age, Cabin and Embarked.
sns.set_style('whitegrid')

sns.countplot(x='Survived', data=train_data)
sns.countplot(x='Survived', data=train_data, hue='Sex')
sns.countplot(x='Survived', data=train_data, hue='Pclass')
sns.distplot(train_data['Age'].dropna(), kde=False, bins=50)
sns.countplot(x='SibSp',data=train_data)
train_data['Fare'].hist(color='green',bins=40,figsize=(8,4))
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=train_data)
#Since Ages of the passengers differs in different classes, so we will fill the ages with mean age with respect to passenger class.
def age_fill(columns):

    age = columns[0]

    pclass = columns[1]

    

    if pd.isnull(age):

        if pclass == 1:

            return int(train_data[train_data['Pclass']==1]['Age'].mean())

        elif pclass == 2:

            return int(train_data[train_data['Pclass']==2]['Age'].mean())

        else:

            return int(train_data[train_data['Pclass']==3]['Age'].mean())

        

    else:

        return age
train_data['Age'] = train_data[['Age','Pclass']].apply(age_fill, axis=1)
sns.heatmap(train_data.isnull(), yticklabels=False, cmap='viridis', cbar=False)
# Lets drop the column Cabin and some null values in Embarked column
train_data.drop('Cabin', axis=1, inplace=True)
train_data.dropna(inplace=True)
train_data.head()
train_data.info()
# Dealing with the categorical values

sex = pd.get_dummies(train_data['Sex'],drop_first=True)

embark = pd.get_dummies(train_data['Embarked'],drop_first=True)
train_data.drop(['Sex','Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
train_data = pd.concat([train_data, sex, embark], axis=1)
train_data.head()
test_data.head()
test_data.info()
test_data['Age'] = test_data[['Age','Pclass']].apply(age_fill, axis=1)
def fill_fare(col):

    pclass = col[0]

    fare = col[1]

    if pd.isnull(fare):

        if pclass == 1:

            return int(train_data[train_data['Pclass']==1]['Fare'].mean())

        elif pclass == 2:

            return int(train_data[train_data['Pclass']==2]['Fare'].mean())

        else:

            return int(train_data[train_data['Pclass']==3]['Fare'].mean())

        

    else:

        return fare
test_data['Fare'] = test_data[['Pclass','Fare']].apply(fill_fare, axis=1)
test_data.info()
test_data.drop('Cabin', axis=1, inplace=True)
test_data.info()
sex = pd.get_dummies(test_data['Sex'],drop_first=True)

embark = pd.get_dummies(test_data['Embarked'],drop_first=True)
test_data.drop(['Sex','Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
test_data.info()
test_data = pd.concat([test_data, sex, embark], axis=1)
train_data.head()
test_data.head()
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ['Pclass','Age','SibSp','Parch','Fare','male','Q','S']

X = train_data[features]

X_test = test_data[features]



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
#Creating Logistic Regression Model
from sklearn.linear_model import LogisticRegression



LogModel = LogisticRegression()

LogModel.fit(X,y)
Logpred = LogModel.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': Logpred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")