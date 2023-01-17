# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd

import pandas_profiling as pdp # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/gender_submission.csv')
df
df.to_csv('submission.csv', index=False)
#Read train data, test data, sample submition data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

gender_submission = pd.read_csv('../input/gender_submission.csv')
train.head(5)
print(train.shape)

print(test.shape)

print(gender_submission.shape)

#Check col_name

print(train.columns)

print('-' * 30) 

print(test.columns)
train.info()
test.info()
train.head()
train.isnull().sum()
test.isnull().sum()
#Connect A and B vertically

df_full = pd.concat([train, test], axis=0, sort=False)

print(df_full.shape)
#Summary statistics

df_full.describe()
df_full.describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8,.9])
#Number of elements, number of items, frequent, frequency of the apearance

df_full.describe(include = 'O')
import pandas_profiling as pdp



#Create report

pdp.ProfileReport(train) 
import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns
#Visualize deaths and survivors

sns.countplot(x='Survived', data=train)

plt.title('Number of deaths and survivors')

plt.xticks([0, 1], ['deaths', 'survivors'])

plt.show()
display(train['Survived'].value_counts())
display(train['Survived'].value_counts()/len(train['Survived']))
#Visualize tne number of survivors by gender

sns.countplot(x='Sex', hue='Survived', data=train)

plt.title('Number of deaths and survivors by gender')

plt.legend(['deathes', 'survivors'])

plt.show()

display(pd.crosstab(train['Sex'], train['Survived']))
#Visualize the number of survivors by tickets class

sns.countplot(x='Pclass', hue='Survived', data=train)

plt.title('Number of deathes and survivors by tikets class')

plt.legend(['deathes', 'survivors'])

plt.show()
display(pd.crosstab(train['Sex'], train['Survived'], normalize='index'))
display(pd.crosstab(train['Pclass'], train['Survived']))
display(pd.crosstab(train['Pclass'], train['Survived'], normalize='index') )
#Visualize the distribution of age

#histgram of the whole

sns.distplot(train['Age'].dropna(),kde=False, bins=30, label='whole')



#histgram of the deathes

sns.distplot(train[train['Survived'] == 0].Age.dropna(),

             kde=False, bins=30, label='deathes')



#histgram of the survivors

sns.distplot(train[train['Survived'] == 1].Age.dropna(),

             kde=False, bins=30, label='survivors')



plt.title('Distribution of age')

plt.legend()

plt.show()
#Divide the age into eight and display the survival rate

train['CateforicAge'] = pd.cut(train['Age'], 8)

display(pd.crosstab(train['CateforicAge'], train['Survived']))
display(pd.crosstab(train['CateforicAge'], train['Survived'], normalize='index'))
#Display the number of brothers and spouses

sns.countplot(x='SibSp', data=train)

plt.title('Number of brothers and spouses')

plt.show()
#Create feature

train['SibSp_0_1_2over'] = [i if i <= 1 else 2 for i in train['SibSp']]
sns.countplot(x='SibSp_0_1_2over', hue='Survived', data=train)

plt.legend(['deathes', 'survivors'])

plt.xticks([0, 1, 2], ['0', '1', '2 or more'])

plt.title('Number of deaths and survivors with brothers and spouses')

plt.show()
display(pd.crosstab(train['SibSp_0_1_2over'], train['Survived']))
display(pd.crosstab(train['SibSp_0_1_2over'], train['Survived'], normalize='index'))
#Visualize the survival and the deathes of parents and children

sns.countplot(x='Parch', data = train)

plt.title('Number of parents and children')

plt.show()
#Create feature

train['Parch_0_1_2_3over'] = [i if i <= 2 else 3 for i in train['Parch']]
sns.countplot(x='Parch_0_1_2_3over', hue='Survived', data=train)

plt.title('Number of deaths and survivors with with parents and childrens')

plt.legend(['deathes', 'survivors'])

plt.xticks([0, 1, 2, 3], ['0', '1', '2', '3or more'])

plt.xlabel('Parch')

plt.show()
display(pd.crosstab(train['Parch_0_1_2_3over'], train['Survived']))
display(pd.crosstab(train['Parch_0_1_2_3over'], train['Survived'], normalize='index'))
#Create and visualize the feature of family

train['FamilySize'] = train['SibSp'] + train['Parch'] +1



train['IsAlone'] = 0

train.loc[train['FamilySize'] >= 2, 'IsAlone'] = 1
sns.countplot(x = 'IsAlone', hue = 'Survived', data = train )

plt.xticks([0, 1], ['1', '2 or more'])

plt.legend(['deathes', 'survivors'])

plt.title('Number of deathed and survivors by 1 or 2 or more')

plt.show()
display(pd.crosstab(train['IsAlone'], train['Survived']))
display(pd.crosstab(train['IsAlone'], train['Survived'], normalize='index'))
#Distribution of fare

sns.distplot(train['Fare'].dropna(), kde=False, hist=True)

plt.title('Distibution of fare')

plt.show()
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], 

                                             as_index=False).mean()
display(pd.crosstab(train['CategoricalFare'], train['Survived']))
display(pd.crosstab(train['CategoricalFare'], train['Survived'], normalize='index'))
#Convert name to the feature

train['Name'][0:5]
#Extract title and eliminate dipulicate

set(train.Name.str.extract('([A-Za-z]+)\.', expand=False))
#Count title

train.Name.str.extract('([A-Za-z]+)\.', expand=False).value_counts()
#Create 'Title' column in train

train['Title'] = train.Name.str.extract('([A-Za-z]+)\.', expand=False)
train.groupby('Title').mean()['Age']
def title_to_num(title):

    if title == 'Master':

        return 1

    elif title == 'Miss':

        return 2

    elif title == 'Mr':

        return 3

    elif title == 'Mrs':

        return 4

    else:

        return 5

            
#Create 'Title' column

test['Title'] = test.Name.str.extract('([A-Za-z]+)\.', expand=False)
train['Title_num'] = [title_to_num(i) for i in train['Title']]

test['Title_num'] = [title_to_num(i) for i in test['Title']]
#Convert categorical variables with 'One-Hot encoding'

#'One-Hot encoding' for Sex and Embotked by pd.get_dummies method

train = pd.get_dummies(train, columns=['Sex', 'Embarked'])

test = pd.get_dummies(test, columns=['Sex', 'Embarked'])
test.head()
train.head(5)
#Remove uannecessary columuns

train.drop(['PassengerId', 'Name', 'Cabin', 'Ticket', 'Title', 'CateforicAge', 'SibSp_0_1_2over', 'Parch_0_1_2_3over', 'FamilySize', 'IsAlone', 'CategoricalFare'], axis=1, inplace=True)

test.drop(['PassengerId', 'Name', 'Cabin', 'Ticket', 'Title'], axis=1, inplace=True)
test.head()
train.head()
X_train = train.drop(['Survived'], axis=1)

y_train = train['Survived']
import lightgbm as lgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
#Divide train and valid

train_X, valid_X, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.33, random_state=0)
gbm = lgb.LGBMClassifier(objective='binary') 
gbm.fit(train_X, train_y, eval_set=[(valid_X, valid_y)], early_stopping_rounds=20, verbose=10);
oof = gbm.predict(valid_X, num_iteration=gbm.best_iteration_)

print('score', round(accuracy_score(valid_y, oof)*100,2))
gbm.predict(valid_X, num_iteration=gbm.best_iteration_)
test_pred = gbm.predict(test, num_iteration=gbm.best_iteration_)
gbm.best_iteration_
gbm.predict(test, num_iteration=gbm.best_iteration_)
gbm.predict_proba(test, num_iteration=gbm.best_iteration_)
test_pred
gender_submission['Survived']=test_pred
gender_submission.to_csv('train_test_split.csv', index=False)
from sklearn.model_selection import KFold



kf = KFold(n_splits=3)



score_list = []

models = []



for fold_,(train_index, valid_index) in enumerate(kf.split(X_train, y_train)):

    train_X = X_train.iloc[train_index]

    valid_X = X_train.iloc[valid_index]

    train_y = y_train[train_index]

    valid_y = y_train[valid_index]

    

    print(f'fold{fold_ + 1} start')

    

    gbm = lgb.LGBMClassifier(objective='binary')

    gbm.fit(train_X, train_y, eval_set = [(valid_X, valid_y)],

           early_stopping_rounds=20,

           verbose= -1)

    

    oof = gbm.predict(valid_X, num_iteration=gbm.best_iteration_)

    score_list.append(round(accuracy_score(valid_y, oof) * 100, 2))

    models.append(gbm)

    print(f'fold{fold_ + 1}end\n')

print(score_list, 'average_score', np.mean(score_list))
#Store predict test data

#Create a 418-by-3 numpy　matrix



test_pred = np.zeros((len(test), 3))
for fold_, gbm in enumerate(models):

    pred_ = gbm.predict(test, num_iteration = gbm.best_iteration_)

    test_pred[:, fold_] = pred_
pred = (np.mean(test_pred, axis=1)>0.5).astype(int)

gender_submission['Survived'] = pred
gender_submission.to_csv('kfpld.csv', index=False)
gbm.get_params()
#Adjust parameters

from sklearn.model_selection import GridSearchCV



gbm = lgb.LGBMClassifier(objecttive='binary')
#List the parameters

params = {

    'max_depth':[2, 3, 4, 5],

    'reg_alpha':[0, 1, 10, 100],

    'reg_lambda':[0, 1, 10, 100],

}
grid_search = GridSearchCV(gbm, param_grid=params, cv=3)
grid_search.fit(X_train, y_train)
print(grid_search.best_score_)
print(grid_search.best_params_)
score = []

test_pred = np.zeros((len(test), 3))
for fold_, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):

    train_X = X_train.iloc[train_index]

    valid_X = X_train.iloc[valid_index]

    train_y = y_train[train_index]

    valid_y = y_train[valid_index]

    

    print(f'fold{fold_ + 1}start')

    gbm = lgb.LGBMClassifier(objective='binary', max_depth=5, reg_alpha=0, reg_lambda=1)

    

    gbm.fit(train_X, train_y,

           eval_set = [(valid_X, valid_y)],

           early_stopping_rounds=20,

           verbose= -1)

    

    oof = gbm.predict(valid_X, num_iteration=gbm.best_iteration_)

    score_list.append(round(accuracy_score(valid_y, oof) * 100, 2))

    test_pred[:, fold_] = gbm.predict(test, num_iteration=gbm.best_iteration_)

    print(f'fold{fold_ + 1} end\n')
print(score_list, 'average_score', np.mean(score_list))
pred = (np.mean(test_pred, axis=1)>0.5).astype(int)

gender_submission['Survived'] = pred
gender_submission.to_csv('glid_search.csv', index=False)