import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head()
train.info()
sns.heatmap(train.isnull(), yticklabels=False, cbar=False)
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Pclass', data=train)
sns.countplot(x='Survived', hue='Sex', data=train)
sns.distplot(train['Age'].dropna(), bins=30)
sns.countplot(x='SibSp', data=train)
sns.distplot(train['Fare'])
sns.boxplot(x='Pclass', y='Age', data=train)
def fillage(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age
train['Age'] = train[['Age', 'Pclass']].apply(fillage, axis=1)

test['Age'] = test[['Age', 'Pclass']].apply(fillage, axis=1)
sns.heatmap(train.isnull(), yticklabels=False, cbar=False)
train.drop('Cabin', axis=1, inplace=True)

test.drop('Cabin', axis=1, inplace=True)
train.dropna(inplace=True)
sns.heatmap(train.isnull(), yticklabels=False, cbar=False)
test.iloc[152,8] = np.mean(test['Fare'])
sex = pd.get_dummies(train['Sex'], drop_first=True)

embarked = pd.get_dummies(train['Embarked'], drop_first=True)

sex_test = pd.get_dummies(test['Sex'], drop_first=True)

embarked_test = pd.get_dummies(test['Embarked'], drop_first=True)
train.drop(['Name', 'Ticket','Sex','Embarked'], inplace=True, axis=1)

test.drop(['Name', 'Ticket','Sex','Embarked'], inplace=True, axis=1)
train = pd.concat([train, sex, embarked], axis=1)

test = pd.concat([test, sex_test, embarked_test], axis=1)
train.head()
from sklearn.linear_model import LogisticRegression
X_train = train.drop('Survived', axis=1)

X_test = test
y_train = train['Survived']
logmodel = LogisticRegression(solver='lbfgs', max_iter=500)
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
test.head()
predictions = pd.DataFrame(predictions, columns=['Survived'])

submission = pd.concat([test['PassengerId'], predictions], axis=1)
submission.head()
submission.to_csv('submission.csv', index=False)