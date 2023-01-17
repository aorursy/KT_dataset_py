# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_list = [df_train, df_test]
print('TRAIN DATASET')

df_train.info()

print('-'*60 + '\n' + '-'*60)

print('TEST DATASET')

df_test.info()
# Datasets Dimensions

df_train.shape, df_test.shape
# Comparing columns of train and test datasets

df_train.columns [ ~df_train.columns.isin(df_test.columns) ]
# Creating dataframe with train and test datasets together for analysis

df = pd.concat([df_train, df_test])
# Features information

df.info()
# Some descriptive statistics metrics of numeric features

df.describe()
# Total number of passgengers (training set)

qtd_survived = len(df_train[df_train['Survived'] == 1])

qtd_notsurvived = len(df_train[df_train['Survived'] == 0])

total_passengers = len(df_train)



print('Total number of passengers: {}\n - Survived: {} ({:.0%})\n - Not survived: {} ({:.0%})'

      .format(total_passengers, qtd_survived, qtd_survived/total_passengers, qtd_notsurvived, qtd_notsurvived/total_passengers))



df_train['Survived'].value_counts().plot(kind='bar')
# Check if feature Sex has any indication of survival rates among passengers.

sns.countplot(x='Survived', data=df_train, hue='Sex', saturation = .4)
# Check if feature Pclass has any indication of survival rates among passengers.

sns.countplot(x='Survived', data=df_train, hue='Pclass', saturation=.4)
# Check if feature Pclass of Sex has any indication of survival rates among passengers.

print(df_train.groupby(['Sex', 'Pclass']).size())

sns.catplot(x='Survived', data=df_train, hue='Pclass', col='Sex', kind='count', saturation=0.5)
# Check if features Age and Sex has any indication of survival rates among passengers.

df_train[(df_train['Age'].notnull()) & (df_train['Sex'] == 'male')]['Age'].plot(kind='hist', alpha=.5, label='male', legend=True)

df_train[(df_train['Age'].notnull()) & (df_train['Sex'] == 'female')]['Age'].plot(kind='hist', alpha=.5, label='female', legend=True)
# Check if features Age and Sex has any indication of survival rates among passengers.

sns.catplot(x='Sex', y='Age', data=df_train, hue='Survived', kind='violin', split=True, saturation=0.5, inner='quartile', scale='count')
sns.catplot(x='Sex', y='Age', data=df_train, hue='Survived', kind='violin', col='Pclass', split=True, saturation=0.5, inner='quartile', scale='count')
print('TRAINING DATASET')

print( df_train.isnull().sum() )

print('-'*60 + '\n' + '-'*60)

print('TEST DATASET')

print( df_test.isnull().sum() )
# Fill NaN values with median of feature age

age_median = df.Age.median()

for dataset in df_list:

    dataset['Age'].fillna(age_median, inplace = True)



# Fill NaN values with mode of feature embarked

embarked_mode = df.Embarked.mode()[0]

for dataset in df_list:

    dataset['Embarked'].fillna(embarked_mode, inplace = True)



# Fill NaN values with median of feature fare

fare_median = df.Fare.median()

df_test['Fare'].fillna(fare_median, inplace = True)



# Fill NaN values with 'Unknown' value

for dataset in df_list:    

    dataset['Cabin'].fillna('Unknown', inplace = True)
print('TRAINING DATASET')

print( df_train.isnull().sum() )

print('-'*60 + '\n' + '-'*60)

print('TEST DATASET')

print( df_test.isnull().sum() )
# Transforming cabin feature in a single character value taking off numeric values

# Transform categorical value to numerical value

map_cabin = {'U':0, 'C':1, 'E':2, 'G':3, 'D':4, 'A':5, 'B':6, 'F':7, 'T':8}

for dataset in df_list:

    dataset['Cabin'] = dataset['Cabin'].str[:1]

    print(dataset['Cabin'].unique())

    dataset['Cabin'] = dataset['Cabin'].map(map_cabin)
# Transform sex feature in numerical value

map_sex = {'male':0, 'female':1}



for dataset in df_list:

    dataset['Sex'] = dataset['Sex'].map(map_sex)
# Transform embarked feature in numerical value

map_embarked = {'S':0, 'C':1, 'Q':2}



for dataset in df_list:

    dataset['Embarked'] = dataset['Embarked'].map(map_embarked)
# Creating family_size feature (SibSp + Parch + 1)

for dataset in df_list:

    dataset['family_size'] = dataset.apply(lambda x: x['SibSp'] + x['Parch'] + 1, axis=1)
# Creating title feature with name feature



print(df.Name.str.split(', ').str[1].str.split('.').str[0].value_counts())



map_title = {'Mr':0, 'Master':1, 'Mrs':2, 'Miss':3, 'Other':4}



for dataset in df_list:

    dataset['Title'] = dataset['Name'].str.split(', ').str[1].str.split('.').str[0]

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    dataset['Title'] = dataset['Title'].replace(['Rev', 'Dr', 'Col', 'Major', 'the Countess', 'Lady', 'Don', 'Capt', 'Sir', 'Dona', 'Jonkheer'], 'Other')

    dataset['Title'] = dataset['Title'].map(map_title)



print(df_train['Title'].value_counts(), df_test['Title'].value_counts())
for dataset in df_list:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)
# Format age feature to int type

for dataset in df_list:

    dataset['Age'] = dataset['Age'].astype(int)
feature_list = ["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked", 'family_size', 'Title', 'Cabin']



data, target = df_train[feature_list], df_train['Survived']

data_test = df_test[feature_list]
knn = KNeighborsClassifier()

knn.fit(data, target)



scores = cross_val_score(knn, data, target, cv=5)

scores
forest = RandomForestClassifier(n_estimators = 1000, random_state=1)

forest.fit(data, target)
forest.score(data, target)
feature_importances = pd.Series(forest.feature_importances_, index=data.columns).sort_values(ascending=False)

print(feature_importances)



scores = cross_val_score(forest, data, target, cv=5, scoring='accuracy')

print(scores, scores.mean())
param_grid = {

    'max_depth': [5, 10, 20, 30],

    'max_features': [2, 3, 5, 10],

    #'min_samples_leaf': [3, 4, 5],

    #'min_samples_split': [8, 10, 12],

    'n_estimators': [100, 200, 300]

}
grid = GridSearchCV(forest, param_grid, scoring='accuracy')
grid.fit(data, target)
grid.best_params_
grid.best_estimator_.predict(data_test)
#pred = forest.predict(data_test)

pred = grid.best_estimator_.predict(data_test)

output = pd.DataFrame({'PassengerId':df_test['PassengerId'],'Survived':pred})

output.to_csv('submission.csv', index=False)