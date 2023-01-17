#importing the necessary libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import accuracy_score

import statsmodels.api as sm

import warnings

warnings.filterwarnings('ignore')
#reading the datasets into dataframes



df_train = pd.read_csv("../input/titanic/train.csv")

df_test = pd.read_csv("../input/titanic/test.csv")
df_train.head(10)
df_train.describe()
df_train.isnull().sum()*100/len(df_train)
#imputing median for missing 'Age' values



med_imputer = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

df_train['Age'] = med_imputer.fit_transform(df_train[['Age']])
#mapping age values to categories



def age_map(x):

    if(x <= 12):

        return "Kid"

    elif(x >= 13 and x <= 19):

        return "Teen"

    elif(x >= 19 and x < 70):

        return "Adult"

    else:

        return "Senior"

df_train['Age'] = df_train['Age'].apply(lambda x: age_map(x))
#processing 'Cabin' and 'Embarked' columns



df_train['Cabin'].fillna('U', inplace=True)

df_train['Cabin'] = df_train['Cabin'].apply(lambda x: x[0])



df_train['Embarked'].fillna('S', inplace=True)
#converting 'Fare' values to categorical



df_train['Fare'] = pd.cut(df_train['Fare'], bins=[-1, 7, 11, 15, 22, 40, 520], labels=[1, 2, 3, 4, 5, 6])
df_train.drop(['Ticket', 'Name'], axis=1, inplace=True)
df_train.head()
fig, ax = plt.subplots(1,2,figsize=(10,5))

sns.countplot(df_train['Age'], data=df_train, ax=ax[0])

sns.countplot(df_train['Age'], hue='Survived', data=df_train, ax=ax[1])

ax[0].title.set_text("Frequency of each age group")

ax[1].title.set_text("Survived: Age Group")
fig, ax = plt.subplots(1,2, figsize=(10,5))

sns.countplot(df_train['Sex'], data=df_train, ax=ax[0])

sns.countplot(df_train['Sex'], hue='Survived', data=df_train, ax=ax[1])

ax[0].title.set_text("Frequency: Sex")

ax[1].title.set_text("Survived: Sex")
fig, ax = plt.subplots(1,2, figsize=(10,5))

sns.countplot(df_train['Pclass'], data=df_train, ax=ax[0])

sns.countplot(df_train['Pclass'], hue='Survived', data=df_train, ax=ax[1])

ax[0].title.set_text("Frequency: Pclass")

ax[1].title.set_text("Survived: Pclass")
fig, ax = plt.subplots(1,2, figsize=(10,5))

sns.countplot(df_train['SibSp'], data=df_train, ax=ax[0])

sns.countplot(df_train['SibSp'], hue='Survived', data=df_train, ax=ax[1])

ax[0].title.set_text("Frequency: SibSp")

ax[1].title.set_text("Survived: SibSp")
fig, ax = plt.subplots(1,2, figsize=(10,5))

sns.countplot(df_train['Embarked'], data=df_train, ax=ax[0])

sns.countplot(df_train['Embarked'], hue='Survived', data=df_train, ax=ax[1])

ax[0].title.set_text("Frequency: Embarked")

ax[1].title.set_text("Survived: Embarked")

fig, ax = plt.subplots(1,2, figsize=(10,5))

sns.countplot(df_train['Parch'], data=df_train, ax=ax[0])

sns.countplot(df_train['Parch'], hue='Survived', data=df_train, ax=ax[1])

ax[0].title.set_text("Frequency: Parch")

ax[1].title.set_text("Survived: Parch")
fig, ax = plt.subplots(1,2, figsize=(10,5))

sns.countplot(df_train['Cabin'], data=df_train, ax=ax[0])

sns.countplot(df_train['Cabin'], hue='Survived', data=df_train, ax=ax[1])

ax[0].title.set_text("Frequency: Cabin")

ax[1].title.set_text("Survived: Cabin")
#LabelEncoder

LE = LabelEncoder()



#label encoding the remaining categorical and continous variables

df_train['Sex'] = LE.fit_transform(df_train['Sex'])

df_train['Cabin'] = LE.fit_transform(df_train['Cabin'])

df_train['Embarked'] = LE.fit_transform(df_train['Embarked'])

df_train['Age'] = LE.fit_transform(df_train['Age'])
#plotting a heatmap of the train set



plt.figure(figsize=(10,10))

sns.heatmap(df_train.corr(), xticklabels = df_train.columns.values, yticklabels = df_train.columns.values, annot=True, cmap="YlGnBu")
df_train.head(10)
#sorting PassendgerId in ascending order

df_train.sort_values(by=['PassengerId'], inplace=True)
#Splitting the train set into dependent and independent variables

y = df_train['Survived']

X = df_train.drop('Survived', axis = 1)



#converting 'Fare' values to int64 type

X['Fare'] = X['Fare'].astype('int64')
#train-test split



X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.65, test_size = 0.35, random_state=100 )
#creating a RandomForestClassifier object and generate the model

model = RandomForestClassifier(n_estimators=100)



# Fit on training data

model.fit(X_train, y_train)
#making predictions on validation set

y_preds = model.predict(X_valid)
df_test.head()
#imputing missing values in 'Age' with the median

age_imputer = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

df_test['Age'] = age_imputer.fit_transform(df_test[['Age']])



#converting age values to categorical values

df_test['Age'] = df_test['Age'].apply(lambda x: age_map(x))



#processing Cabin and Embarked columns

df_test['Cabin'].fillna('U', inplace=True)

df_test['Cabin'] = df_test['Cabin'].apply(lambda x: x[0])



df_test['Embarked'].fillna('S', inplace=True)



#imputing missing values in 'Fare' with the mean

fare_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

df_test['Fare'] = fare_imputer.fit_transform(df_test[['Fare']])



#converting 'Fare' values to categorical

df_test['Fare'] = pd.cut(df_test['Fare'], bins=[-1, 7, 11, 15, 22, 40, 520], labels=[1, 2, 3, 4, 5, 6])



#label encoding the remaining categorical and continous variables

df_test['Sex'] = LE.fit_transform(df_test['Sex'])

df_test['Cabin'] = LE.fit_transform(df_test['Cabin'])

df_test['Embarked'] = LE.fit_transform(df_test['Embarked'])

df_test['Age'] = LE.fit_transform(df_test['Age'])





#converting 'Fare' values to int64 type

df_test['Fare'] = df_test['Fare'].astype('int64')



#dropping Name and Ticket columns from the test set

df_test.drop(['Name','Ticket'], axis=1, inplace=True)
df_test.head()
#sorting PassendgerId in ascending order

df_test.sort_values(by=['PassengerId'], inplace=True)
#making predictions on the test set

y_test_pred = model.predict(df_test)
#write to output file

output = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_test_pred})

output.to_csv("submission.csv", index=False)