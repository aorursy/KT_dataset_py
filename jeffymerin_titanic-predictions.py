# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Data Visualization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

y_sample =  pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
train_data.head(10)
train_data.info()
import seaborn as sns

sns.countplot(x = "Survived",data = train_data)
train_data.groupby('Survived').mean()
train_data.describe()
train_data.groupby(train_data['Age'].isnull()).mean()
age_survived = train_data[train_data['Survived'] == 1]['Age'].dropna()

age_died = train_data[train_data['Survived'] == 0]['Age'].dropna()

sns.distplot(age_survived, kde = False, bins = 40)

sns.distplot(age_died, kde = False, bins = 40)

plt.legend(['Did not survive', 'Survived'])

plt.title('Overlaid histogram for Age')

plt.show()
fare_survived = train_data[train_data['Survived'] == 1]['Fare'].dropna()

fare_died = train_data[train_data['Survived'] == 0]['Fare'].dropna()

sns.distplot(fare_survived, kde = False, bins = 20)

sns.distplot(fare_died, kde = False, bins = 20)

plt.legend(['Did not survive', 'Survived'])

plt.title('Overlaid histogram for Fare')

plt.show()
sns.heatmap(train_data[['Fare', 'Pclass']].corr(),annot = True)
train_data.groupby('Pclass').mean()
sns.catplot(x='Pclass', y='Survived', data=train_data, kind='point', aspect=2)
sns.catplot(x='SibSp', y='Survived', data=train_data, kind='point', aspect=2)
sns.catplot(x='Parch', y='Survived', data=train_data, kind='point', aspect=2)
import seaborn as sns

sns.boxplot(x = 'Parch', y = 'Age', data = train_data, palette = 'hls')
train_data.groupby('Parch').mean()
def age_aprox(cols):

    age = cols[0]

    parch = cols[1]

    

    if pd.isnull(age):

        if parch == 0:

            return train_data[train_data['Parch'] == 0]['Age'].mean()

        elif parch == 1:

            return train_data[train_data['Parch'] == 1]['Age'].mean()

        elif parch == 2:

            return train_data[train_data['Parch'] == 2]['Age'].mean()

        elif parch == 3:

            return  train_data[train_data['Parch'] == 3]['Age'].mean()

        elif parch == 4:

            return train_data[train_data['Parch'] == 4]['Age'].mean()

        else:

            return train_data['Age'].mean()

    else:

        return age
train_data['Age'] = train_data[['Age','Parch']].apply(age_aprox, axis = 1)

test_data['Age'] = test_data[['Age','Parch']].apply(age_aprox, axis = 1)
train_data['Family_count'] = train_data['SibSp'] +train_data['Parch']

#train_data.drop(['SibSp','Parch'], inplace = True,axis  = 1)

test_data['Family_count'] = test_data['SibSp'] +test_data['Parch']

#test_data.drop(['SibSp','Parch'], inplace = True,axis  = 1)
round((train_data['Cabin'].isnull().sum()/len(train_data))*100,2)
sns.catplot(x='Sex', y='Survived', data=train_data, kind='point', aspect=2)
sns.catplot(x='Embarked', y='Survived', data=train_data, kind='point', aspect=2)
train_data.pivot_table('Survived',index = 'Sex',columns = 'Embarked',aggfunc= 'count')
train_data['Name'].head(10)
# Apply regex per name

# Use function : Series.str.extract()

for name in train_data['Name']:

    train_data['Title'] = train_data['Name'].str.extract('([A-Za-z]+)\.',expand=True)    # Regex to get title : ([A-Za-z]+)\.

    test_data['Title'] = test_data['Name'].str.extract('([A-Za-z]+)\.',expand=True) 
train_data.groupby('Title').count()['PassengerId']
test_data.groupby('Title').count()['PassengerId']
gender = {'male':1,'female': 0}

train_data['Sex'] = train_data['Sex'].map(gender)

test_data['Sex'] = test_data['Sex'].map(gender)
train_data.head(5)
train_data['Embarked'].describe()
train_data['Embarked'].fillna('S', inplace = True)

test_data['Embarked'].fillna('S', inplace = True)
title = {'Capt':'Others','Col':'Others','Countess':'Others','Don':'Others', 'Dr':'Others','Jonkheer':'Others', 'Lady':'Others', 'Major':'Others',

        'Mlle':'Others', 'Mme':'Others', 'Ms':'Miss','Rev': 'Others','Sir':'Others','Dona': 'Others'}
train_data.replace({'Title':title},inplace=True)

test_data.replace({'Title':title},inplace=True)
train_data.groupby('Title').count()['PassengerId']
features = ["Pclass", "Sex", "SibSp","Parch","Age","Embarked"]

y = train_data["Survived"]

X = pd.get_dummies(train_data[features])

titanic_test = pd.get_dummies(test_data[features])

print(X.shape)

print(y.shape)

#print(X_test.shape)
sns.heatmap(X.corr(),annot = True)
from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
for dataset in [y_train,y_val,y_test]:

    print(round(len(dataset)/len(y),2))
from sklearn.model_selection import  cross_val_score
from sklearn.ensemble import RandomForestClassifier

rf_basic = RandomForestClassifier()

scores  = cross_val_score(rf_basic,X_train,y_train, cv = 5)

scores
from pprint import pprint

# Look at parameters used by our current forest

print('Parameters currently in use:\n')

pprint(rf_basic.get_params())
from sklearn.model_selection import GridSearchCV
def print_results(results):

    print('BEST PARAMS: {}\n'.format(results.best_params_))



    means = results.cv_results_['mean_test_score']

    stds = results.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, results.cv_results_['params']):

        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))
rf_grid = RandomForestClassifier()

parameters = {

    'n_estimators': [5, 50, 100, 200],

    'max_depth': [2, 10, 20, None],

    'max_features': ['auto', 'sqrt'],

    'min_samples_leaf': [1, 2, 4],

    'min_samples_split': [2, 5, 10],

}



cv = GridSearchCV(rf_grid, parameters, cv=5,verbose = 2)

#cv.fit(X_train, y_train)
#print_results(cv)
from sklearn.metrics import accuracy_score, precision_score, recall_score
rf_best = RandomForestClassifier(n_estimators= 50, max_depth = 10, max_features = 'auto', min_samples_leaf =4, min_samples_split = 10)

rf_best.fit(X_train,y_train)
mdl = rf_best

y_pred = mdl.predict(X_val)

accuracy = round(accuracy_score(y_val, y_pred), 3)

precision = round(precision_score(y_val, y_pred), 3)

recall = round(recall_score(y_val, y_pred), 3)

print('MAX DEPTH: {} / # OF EST: {} -- A: {} / P: {} / R: {}'.format(mdl.max_depth,

                                                                         mdl.n_estimators,

                                                                         accuracy,

                                                                         precision,

                                                                         recall))
y_pred = rf_best.predict(X_test)

accuracy = round(accuracy_score(y_test, y_pred), 3)

precision = round(precision_score(y_test, y_pred), 3)

recall = round(recall_score(y_test, y_pred), 3)

print('MAX DEPTH: {} / # OF EST: {} -- A: {} / P: {} / R: {}'.format(rf_best.max_depth,

                                                                     rf_best.n_estimators,

                                                                     accuracy,

                                                                     precision,

                                                                     recall))
from sklearn.ensemble import RandomForestClassifier

rf_best.fit(X, y)
predictions = rf_best.predict(titanic_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
output