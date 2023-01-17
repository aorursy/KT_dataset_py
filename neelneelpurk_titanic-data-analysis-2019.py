# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import warnings

warnings.filterwarnings("ignore")



import pandas as pd

import numpy as np

import collections

import re

import copy



#from pandas.tools.plotting import scatter_matrix

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('bmh')



%matplotlib inline



from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier, plot_importance

from sklearn.model_selection import cross_val_score



from sklearn.model_selection import GridSearchCV



pd.set_option('display.max_columns', 500)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.info()
train.head(5)
train.describe()
train.describe(include=['O'])
## exctract cabin letter

def extract_cabin(x):

    return x!=x and 'other' or x[0]

train['Cabin_l'] = train['Cabin'].apply(extract_cabin)

train.head(5)
plain_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Cabin_l']

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

start = 0

for j in range(2):

    for i in range(3):

        if start == len(plain_features):

            break

        sns.barplot(x=plain_features[start],

                    y='Survived', data=train, ax=ax[j, i])

        start += 1
sv_lab = 'survived'

nsv_lab = 'not survived'

fig, ax = plt.subplots(figsize=(5, 3))

ax = sns.distplot(train[train['Survived'] == 1].Age.dropna(),

                  bins=20, label=sv_lab, ax=ax)

ax = sns.distplot(train[train['Survived'] == 0].Age.dropna(),

                  bins=20, label=nsv_lab, ax=ax)

ax.legend()

ax.set_ylabel('KDE');



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

females = train[train['Sex'] == 'female']

males = train[train['Sex'] == 'male']



ax = sns.distplot(females[females['Survived'] == 1].Age.dropna(

), bins=30, label=sv_lab, ax=axes[0], kde=False)

ax = sns.distplot(females[females['Survived'] == 0].Age.dropna(

), bins=30, label=nsv_lab, ax=axes[0], kde=False)

ax.legend()

ax.set_title('Female')

ax = sns.distplot(males[males['Survived'] == 1].Age.dropna(),

                  bins=30, label=sv_lab, ax=axes[1], kde=False)

ax = sns.distplot(males[males['Survived'] == 0].Age.dropna(),

                  bins=30, label=nsv_lab, ax=axes[1], kde=False)

ax.legend()

ax.set_title('Male');
sns.catplot('Pclass', 'Survived', hue='Sex', col = 'Embarked', data=train, kind='point');

sns.catplot('Pclass', 'Survived', col = 'Embarked', data=train, kind='point');

ax = sns.boxplot(x="Pclass", y="Fare", hue="Survived", data=train)

ax.set_yscale('log')
sns.violinplot(x='Pclass', y='Age', hue='Survived', data=train, split=True);
# To get the full family size of a person, added siblings and parch.

train['family_size'] = train['SibSp'] + train['Parch'] + 1

test['family_size'] = test['SibSp'] + test['Parch'] + 1

axes = sns.catplot('family_size',

                   'Survived',

                   hue='Sex',

                   data=train,

                   aspect=4,

                   kind='point')
train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

print(collections.Counter(train['Title']).most_common())

test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

print()

print(collections.Counter(test['Title']).most_common())
tab = pd.crosstab(train['Title'],train['Pclass'])

print(tab)

tab_prop = tab.div(tab.sum(1).astype(float), axis=0)

tab_prop.plot(kind="bar", stacked=True)
sns.catplot('Title', 'Survived', data=train, aspect=3, kind='point');
train['Title'].replace(['Master', 'Major', 'Capt', 'Col','Don', 'Sir', 'Jonkheer', 'Dr'], 'titled', inplace=True)

#train['Title'].replace(['Countess','Dona','Lady'], 'titled_women', inplace = True)

#train['Title'].replace(['Master','Major', 'Capt', 'Col','Don', 'Sir', 'Jonkheer', 'Dr'], 'titled_man', inplace = True)

train['Title'].replace(['Countess', 'Dona', 'Lady'], 'Mrs', inplace=True)

#train['Title'].replace(['Master'], 'Mr', inplace = 'True')

train['Title'].replace(['Mme'], 'Mrs', inplace=True)

train['Title'].replace(['Mlle', 'Ms'], 'Miss', inplace=True)
sns.catplot('Title', 'Survived', data=train, aspect=3, kind='point');
def extract_cabin(x):

    return x != x and 'other' or x[0]





train['Cabin_l'] = train['Cabin'].apply(extract_cabin)

print(train.groupby('Cabin_l').size())

sns.catplot('Cabin_l', 'Survived',

            order=['other', 'A', 'B', 'C', 'D', 'E', 'F', 'T'],

            aspect=3,

            data=train,

            kind='point')
plt.figure(figsize=(8, 8))

corrmap = sns.heatmap(train.drop('PassengerId',axis=1).corr(), square=True, annot=True)
train.shape[0] - train.dropna().shape[0]
train.isnull().sum()
max_emb = np.argmax(train['Embarked'].value_counts())

train['Embarked'].fillna(max_emb, inplace=True)
ages = train['Age'].dropna()

std_ages = ages.std()

mean_ages = ages.mean()

train_nas = np.isnan(train["Age"])

test_nas = np.isnan(test["Age"])

np.random.seed(122)

impute_age_train  = np.random.randint(mean_ages - std_ages, mean_ages + std_ages, size = train_nas.sum())

impute_age_test  = np.random.randint(mean_ages - std_ages, mean_ages + std_ages, size = test_nas.sum())

train["Age"][train_nas] = impute_age_train

test["Age"][test_nas] = impute_age_test

ages_imputed = np.concatenate((test["Age"],train["Age"]), axis = 0)
train['Age*Class'] = train['Age']*train['Pclass']

test['Age*Class'] = test['Age']*test['Pclass']
sns.kdeplot(ages_imputed, label = 'After imputation');

sns.kdeplot(ages, label = 'Before imputation');
train_label = train['Survived']

test_pasId = test['PassengerId']

drop_cols = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'PassengerId']

train.drop(drop_cols + ['Cabin_l'], 1, inplace=True)

test.drop(drop_cols, 1, inplace=True)
train['Pclass'] = train['Pclass'].apply(str)

test['Pclass'] = test['Pclass'].apply(str)
#train.drop(['Survived'], 1, inplace=True)

train_objs_num = len(train)

dataset = pd.concat(objs=[train, test], axis=0)

dataset = pd.get_dummies(dataset)

train = copy.copy(dataset[:train_objs_num])

test = copy.copy(dataset[train_objs_num:])
test.head(5)
droppings = ['Embarked_Q','Age']

#droppings += ['Sex_male', 'Sex_female']



test.drop(droppings, 1, inplace=True)

train.drop(droppings, 1, inplace=True)
train.head(5)
test.head(5)
#train.to_csv('train1.csv',index=False)
#test.to_csv('test1.csv',index=False)
def prediction(model, train, label, test, test_pasId):

    model.fit(train, label)

    pred = model.predict(test)

    accuracy = cross_val_score(model, train, label, cv=5)



    sub = pd.DataFrame({

        "PassengerId": test_pasId,

        "Survived": pred

    })

    return [accuracy, sub]
xgb = XGBClassifier(n_estimators=200)

acc_xgb, sub = prediction(xgb, train, train_label, test, test_pasId)

print(acc_xgb)

plot_importance(xgb)
train.head(10)
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(train, train_label)
print('Probability of each class')

print('Survive = 0: %.2f' % classifier.class_prior_[0])

print('Survive = 1: %.2f' % classifier.class_prior_[1])