import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt

import seaborn as sns # easy visualization

%matplotlib inline
# load the test and train sets, concat them together for cleaning and feature engineering

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_full = pd.concat([df_train, df_test], axis = 0, ignore_index=True)



# save the test PassengerID for future use

passengerId_test = df_test['PassengerId']

df_full.head()
df_full.describe()
df_full.info()
plt.figure(figsize=(10,7))

sns.heatmap(df_full.isnull(), yticklabels=False, cbar=False)
sns.set_style('whitegrid')

sns.countplot(x='Survived', data=df_full)
sns.countplot(x='Survived', hue='Sex', data=df_full)
sns.countplot(x='Survived', hue='Pclass', data=df_full)
sns.distplot(df_full['Age'].dropna(), kde=False, bins = 30)
sns.countplot(x='SibSp', data = df_full)
plt.figure(figsize=(10,5))

sns.distplot(df_full['Fare'].dropna(), rug=True, hist=False)
plt.figure(figsize=(10,7))

sns.boxplot(x='Pclass', y='Age', data=df_full)
df_full.groupby('Pclass').mean()['Age']
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        

        if Pclass == 1:

            return 39.159930

        elif Pclass == 2:

            return 29.506705

        else:

            return 24.816367

    

    else:

        return Age
# now apply this function

df_full['Age'] = df_full[['Age', 'Pclass']].apply(impute_age, axis = 1)
df_full.info()
plt.figure(figsize=(10,7))

sns.boxplot(x='Pclass', y='Fare', data=df_full)
df_full.groupby('Pclass').mean()['Fare']
df_full[df_full['Fare'].isnull()]
df_full.loc[df_full['PassengerId'] == 1044, 'Fare'] = 13.302889

df_full[df_full['PassengerId'] == 1044]
df_full[df_full['Embarked'].isnull()]
df_full.loc[df_full['PassengerId'].isin([62, 830]), 'Embarked'] = 'C'

df_full.loc[df_full['PassengerId'].isin([62, 830])]
df_full.info()
df_full.drop(['Cabin', 'Ticket'], axis=1, inplace=True)
df_full.info()
sex = pd.get_dummies(df_full['Sex'], drop_first=True)

sex.head()
embark = pd.get_dummies(df_full['Embarked'], drop_first=True)

embark.head()
df_full = pd.concat([df_full, sex, embark], axis = 1)

df_full.drop(['Sex', 'Embarked'], axis = 1, inplace=True)

df_full.head()
df_full.drop(['Name', 'PassengerId'], axis=1, inplace=True)
df_full.head()
df_train = df_full[:891]

df_test = df_full[891:]
X = df_train.drop('Survived', axis=1)

y = df_train['Survived']
# sklearn imports

from sklearn.cross_validation import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
skf = StratifiedKFold(y, n_folds=3)
for train_index, test_index in skf:

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    logmodel = LogisticRegression()

    logmodel.fit(X_train, y_train)

    predictions = logmodel.predict(X_test)

    print(classification_report(y_test, predictions))
X_test = df_test.drop('Survived', axis=1)

logmodel = LogisticRegression()

logmodel.fit(X, y)

predictions = logmodel.predict(X_test)
df_predictions = pd.DataFrame({'PassengerID' : passengerId_test, 'Survived' : predictions.astype(int)})

df_predictions.to_csv('logistic_regression_submission.csv', index=False)