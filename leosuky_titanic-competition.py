import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()
train = pd.read_csv('/kaggle/input/titanic/train.csv')

#train.drop('Cabin', axis=1, inplace=True)

train.head(2)
test = pd.read_csv('/kaggle/input/titanic/test.csv')

test.head(2)
print(train.shape)

train.isnull().sum()
print(test.shape)

test.isnull().sum()
train.drop('Cabin', axis=1, inplace=True)

test.drop('Cabin', axis=1, inplace=True)
train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.')

test['Title'] = test.Name.str.extract(' ([A-Za-z]+)\.')

print(test.Title.value_counts()) 

print(train.Title.value_counts())
null_age_train = train[train['Age'].isnull()]

null_age_test = test[test['Age'].isnull()]

print(null_age_test.Title.value_counts())

print(null_age_train.Title.value_counts())
age_df = pd.concat([train[['Age', 'Title']], test[['Age', 'Title']]], axis=0)

nnaa = age_df[age_df['Age'].notnull()]

nnaa.shape
median_ages = nnaa[['Age', 'Title']].groupby(['Title'], as_index=False).median()

median_ages.set_index('Title', inplace=True)

print(median_ages.loc['Mr', 'Age'])

median_ages
for title in train.Title:

    if title == 'Mr':

        train['Age'].fillna(value=median_ages.loc['Mr', 'Age'], inplace=True)

    elif title == 'Miss':

        train['Age'].fillna(value=median_ages.loc['Miss', 'Age'], inplace=True)

    elif title == 'Mrs':

        train['Age'].fillna(value=median_ages.loc['Mrs', 'Age'], inplace=True)

    elif title == 'Master':

        train['Age'].fillna(value=median_ages.loc['Master', 'Age'], inplace=True)

    else:

        train['Age'].fillna(value=median_ages.loc['Dr', 'Age'], inplace=True)
for title in test.Title:

    if title == 'Mr':

        test['Age'].fillna(value=median_ages.loc['Mr', 'Age'], inplace=True)

    elif title == 'Miss':

        test['Age'].fillna(value=median_ages.loc['Miss', 'Age'], inplace=True)

    elif title == 'Mrs':

        test['Age'].fillna(value=median_ages.loc['Mrs', 'Age'], inplace=True)

    elif title == 'Master':

        test['Age'].fillna(value=median_ages.loc['Master', 'Age'], inplace=True)

    else:

        test['Age'].fillna(value=median_ages.loc['Miss', 'Age'], inplace=True)
train['Embarked'].fillna(value='S', inplace=True)

(test[['Pclass','Age', 'Fare', 'Embarked']]).sort_values(by='Age', ascending=False).head(18)

test[['Pclass', 'Fare']].groupby(['Pclass'], as_index=False).median()

test['Fare'].fillna(value=7.8958, inplace=True)

test.loc[152, :]
print(train.isnull().sum())

print(test.isnull().sum())
for title in train.Title:

    train['Title'] = train['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Mlle', 'Jonkheer', 'Countess', 'Lady', 'Mme',

                                            'Sir', 'Don', 'Capt'], 'Other')

    train['Title'] = train['Title'].replace('Ms', 'Miss')

    

for title in test.Title:

    test['Title'] = test['Title'].replace(['Dr', 'Rev', 'Col', 'Dona',], 'Other')

    test['Title'] = test['Title'].replace('Ms', 'Miss')
#Correlation between Sex and Survived

print(train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())

sns.barplot(x='Sex', y='Survived', data=train)
#Correlation between Pclass and Survived

print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())

sns.barplot(x='Pclass', y='Survived', data=train)
#Correlation between Embarked and Survived

print(train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())

sns.barplot(x='Embarked', y='Survived', data=train)
#Correlation between Parch and Survived

print(train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean())

sns.barplot(x='Parch', y='Survived', data=train)
#Correlation between SibSp and Survived

print(train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean())

sns.barplot(x='SibSp', y='Survived', data=train)
#Correlation between Title and Survived

print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

sns.barplot(x='Title', y='Survived', data=train)
#Correlation between [Pclass & Sex] and Survived

p_sex = pd.crosstab(train['Pclass'], train['Sex'])

print(p_sex)

sns.factorplot('Sex', 'Survived', hue='Pclass', height=4, aspect=2, data=train)
#selecting the features to train model with

train = train.drop(['PassengerId', 'Name', 'SibSp', 'Ticket'], axis=1)

test = test.drop(['Name', 'SibSp', 'Ticket'], axis=1)



#now we map the object dtypes of Sex, Embarked and Title

train['Sex'] = train['Sex'].map({'female': 1, 'male': 0}).astype(int)

train['Embarked'] = train['Embarked'].map({'S':0, 'Q':1, 'C':2}).astype(int)

train['Title'] = train['Title'].map({'Mr':0, 'Mrs':1, 'Miss':2, 'Master':3, 'Other':4}).astype(int)



#test dataset

test['Sex'] = test['Sex'].map({'female': 1, 'male': 0}).astype(int)

test['Embarked'] = test['Embarked'].map({'S':0, 'Q':1, 'C':2}).astype(int)

test['Title'] = test['Title'].map({'Mr':0, 'Mrs':1, 'Miss':2, 'Master':3, 'Other':4}).astype(int)

a_age_fare = train[['Age', 'Fare']]

b_age_fare = test[['Age', 'Fare']]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_train = scaler.fit_transform(a_age_fare)

new_aaf = pd.DataFrame(scaled_train, index=a_age_fare.index, columns=a_age_fare.columns)



scaled_test = scaler.fit_transform(b_age_fare)

new_baf = pd.DataFrame(scaled_test, index=b_age_fare.index, columns=b_age_fare.columns)
train['Age'] = new_aaf['Age']

train['Fare'] = new_aaf['Fare']

train.head(2)
test['Age'] = new_baf['Age']

test['Fare'] = new_baf['Fare']

test.head(2)
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
X_train = train.drop('Survived', axis=1)

y_train = train['Survived']

X_test = test.drop('PassengerId', axis=1).copy()



X_train.shape, X_test.shape, y_train.shape
#Logistic Regression

Log_model = LogisticRegression(solver='lbfgs')

Log_model.fit(X_train, y_train)

y_pred_Log = Log_model.predict(X_test)

Log_reg_accuracy = round(Log_model.score(X_train, y_train) * 100, 2)

print(str(Log_reg_accuracy) + '%')
#SVC Model

svc_model = SVC(gamma='auto')

svc_model.fit(X_train, y_train)

y_pred_svc = svc_model.predict(X_test)

svc_accuracy = round(svc_model.score(X_train, y_train) * 100, 2)

print(str(svc_accuracy) + '%')
dtc = DecisionTreeClassifier()

gs_dtc = GridSearchCV(dtc,

                 {'max_depth': range(1, 10),

                 'min_samples_split': range(5, 51, 5)},

                 cv=5,

                 n_jobs=2)

gs_dtc.fit(X_train, y_train)

print(gs_dtc.best_params_)
dtc_model = gs_dtc.best_estimator_

dtc_model.fit(X_train, y_train)

y_dtc = dtc_model.predict(X_test)

y_dtc_accuracy = round(dtc_model.score(X_train, y_train) * 100, 2)

print(str(y_dtc_accuracy) + '%')
submission = pd.DataFrame({'PassengerId': test['PassengerId'],

                          'Survived': y_pred_svc})



submission.to_csv('Final_submission.csv', index=False)

#submission.head(4)