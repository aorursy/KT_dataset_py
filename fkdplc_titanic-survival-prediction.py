import numpy as np

import pandas as pd

import os

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.exceptions import ConvergenceWarning

import warnings



warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.simplefilter(action='ignore', category=ConvergenceWarning)

        

PATH = '/kaggle/input/titanic/'



train = pd.read_csv(PATH + 'train.csv')

test = pd.read_csv(PATH + 'test.csv')



train.isna().sum() + test.isna().sum()
# maps columns

def map_dfs(col, mapping):

    train[col] = train[col].map(mapping)

    test[col] = test[col].map(mapping)



# apply func to columns for obtaining new one

def apply_to_dfs(col, new_col, func):

    train[new_col] = train[col].apply(func)

    test[new_col] = test[col].apply(func)



# fill NaNs with mean

def fill_mean(feat):

    train[feat].fillna(train[feat].mean(), inplace=True)

    test[feat].fillna(test[feat].mean(), inplace=True)

    

# drop column

def drop(col, only_train=False):

    train.drop(columns=[col], inplace=True)

    if not only_train:

        test.drop(columns=[col], inplace=True)



# perform OHE

def make_dummies(feat):

    global train

    global test

    

    dum_train = pd.get_dummies(train[feat], prefix=feat, drop_first=True)

    dum_test = pd.get_dummies(test[feat], prefix=feat, drop_first=True)

    

    common_columns = [i for i in dum_train.columns if i in dum_test.columns]

    remove_train = [i for i in dum_train.columns if i not in common_columns]

    remove_test = [i for i in dum_test.columns if i not in common_columns]

    

    train = pd.concat([train, dum_train[common_columns]], axis=1)

    train.drop(columns=[feat], inplace=True)

    test = pd.concat([test, dum_test[common_columns]], axis=1)

    test.drop(columns=[feat], inplace=True)

    

# plot histogram of feature according to target variable

# def plot_feature(feat):

#     plt.xlabel(feat)

#     sns.distplot(train[train['Survived'] == 1][feat], kde=False, label='alive')

#     sns.distplot(train[train['Survived'] == 0][feat], kde=False, label='dead')

#     plt.legend()

    

# plot correlation matrix

def plot_corr(df):

    plt.figure(1, figsize=(16, 16))

    sns.heatmap(df.corr(), annot=True, fmt='.2f', cbar=False)

    plt.show()
plt.figure(1, figsize=(20, 6))

plt.subplot(121)

sns.violinplot(x='Pclass', y='Age', hue='Sex', data=train, split=True)



plt.subplot(122)

sns.violinplot(x='Pclass', y='Fare', hue='Sex', data=train, split=True)

plt.show()
for f in ['Age', 'Fare']:

    train[f] = train.groupby(['Pclass', 'Sex'])[f].transform(lambda x: x.fillna(x.median()))

    test[f] = test.groupby(['Pclass', 'Sex'])[f].transform(lambda x: x.fillna(x.median()))
train = train[((train['Fare'] < 350) & (train['Pclass'] == 1)) |\

              ((train['Fare'] < 50) & (train['Pclass'] == 2)) |\

              ((train['Fare'] < 40) & (train['Pclass'] == 3))] # taken visually from previous plot



# make distribution more normal

train['Fare'] = np.log1p(train['Fare'])

test['Fare'] = np.log1p(test['Fare'])



plt.figure(1, figsize=(20, 6))

plt.subplot(121)

sns.violinplot(x='Pclass', y='Age', hue='Sex', data=train, split=True)



plt.subplot(122)

sns.violinplot(x='Pclass', y='Fare', hue='Sex', data=train, split=True)

plt.show()
train = train[train['Fare'] > 2]
plt.figure(1, figsize=(10, 6))

sns.violinplot(x='Pclass', y='Embarked', hue='Sex', data=train, split=True)

plt.show()
train['Embarked'] = train.groupby(['Pclass', 'Sex'])['Embarked'].transform(lambda x: x.fillna(x.mode()[0]))

test['Embarked'] = test.groupby(['Pclass', 'Sex'])['Embarked'].transform(lambda x: x.fillna(x.mode()[0]))

make_dummies('Embarked')
make_dummies('Pclass')
make_dummies('Sex')
drop('Ticket')
train['Cabin'].value_counts()
apply_to_dfs('Cabin', 'cabin_code', lambda c: c[0] if c == c else '??')

make_dummies('cabin_code')

drop('Cabin')
train['fs'] = train['SibSp'] + train['Parch']

test['fs'] = test['SibSp'] + test['Parch']



apply_to_dfs('fs', 'alone', lambda fs: 1 if fs == 0 else 0)

apply_to_dfs('fs', 'big_family', lambda fs: 1 if fs > 3 else 0)



for col in ['SibSp', 'Parch', 'fs']:

    drop(col)
def extract_last_name(name):

    return name.split(',')[0] if ',' in name else '?'



def extract_addr_form(name):

    if ',' not in name:

        return '?'

    else:

        after_comma = name.split(',')[1][1:]

        return after_comma.split('.')[0] if '.' in after_comma else '?'



def extract_rest_name(name):

    return name.split('.')[1] if '.' in name else '?'



# extract last_name and addr_form

apply_to_dfs('Name', 'last_name', extract_last_name)

apply_to_dfs('Name', 'addr_form', extract_addr_form)



# temporarily extract rest_name

apply_to_dfs('Name', 'rest_name', extract_rest_name)

apply_to_dfs('rest_name', 'name_words_count', lambda name: len(name.split()) + 1)

apply_to_dfs('rest_name', 'has_brakets', lambda name: 1 if '(' in name else 0)

apply_to_dfs('rest_name', 'has_quot', lambda name: 1 if '"' in name else 0)

drop('rest_name')



train[['last_name', 'addr_form', 'name_words_count', 'has_brakets', 'has_quot']].head(20)
last_names = set([i for i in train['last_name']])

'%d unique last names' % len(last_names)
fam_surv_rate = train.groupby(['last_name'])['Survived'].agg(['sum', 'count'])

fam_surv_rate['survival_rate'] = fam_surv_rate['sum'] / fam_surv_rate['count']

fam_surv_rate
# create new columns with 0.5 ('Name' is for stub and not used)

apply_to_dfs('Name', 'survival_rate', lambda n: 0.5)



for i in range(train.shape[0]):

    last_name = train.iloc[i]['last_name']

    if last_name in fam_surv_rate.index:

        train.at[i, 'survival_rate'] = fam_surv_rate.loc[last_name]['survival_rate']

        

for i in range(test.shape[0]):

    last_name = test.iloc[i]['last_name']

    if last_name in fam_surv_rate.index:

        test.at[i, 'survival_rate'] = fam_surv_rate.loc[last_name]['survival_rate']

        

train.dropna(inplace=True)

test.dropna(inplace=True)

        

plt.figure(1, figsize=(10, 6))

sns.violinplot(x='Survived', y='survival_rate', data=train)

plt.show()
plt.figure(1, figsize=(16, 12))



plt.subplot(221)

sns.violinplot(x='name_words_count', y='Survived', data=train)



plt.subplot(222)

sns.violinplot(x='addr_form', y='Survived', data=train)



plt.subplot(223)

sns.violinplot(x='has_brakets', y='Survived', data=train)



plt.subplot(224)

sns.violinplot(x='has_quot', y='Survived', data=train)



plt.show()
make_dummies('addr_form')

apply_to_dfs('name_words_count', 'long_name', lambda c: 1 if c >= 5 else 0)



for col in ['name_words_count', 'last_name', 'Name']:

    drop(col)
# extract target and test_ids for future submission 

target = train['Survived']

test_ids = test['PassengerId']



drop('PassengerId')

drop('Survived', only_train=True)



train.head()
plot_corr(train)
from sklearn.decomposition import PCA



pca = PCA(0.995, random_state=289)

pca.fit(train)



train = pca.transform(train)

test = pca.transform(test)



print('explained variance:', pca.explained_variance_ratio_)
from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split



scaler = RobustScaler().fit(train)

train = scaler.transform(train)

test = scaler.transform(test)



x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=289)



x_train.shape, x_test.shape, y_train.shape, y_test.shape
from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score as acc



def perform_search(estimator, params, extended=False):

    search = GridSearchCV(estimator, params, cv=5)

    search.fit(x_train, y_train)

    results = pd.DataFrame()

    for k, v in search.cv_results_.items():

        results[k] = v

    results = results.sort_values(by='rank_test_score')

    best_params_row = results[results['rank_test_score'] == 1]

    mean, std = best_params_row['mean_test_score'].iloc[0], best_params_row['std_test_score'].iloc[0]

    best_params = best_params_row['params'].iloc[0]

    if extended:

        return best_params, mean, std, results

    else:

        return best_params



depths = [i for i in range(2, 9)]

n_estimators = [50, 75, 100, 125, 150, 200, 250, 300, 400]

    

models = [

    (XGBClassifier(), {

        'max_depth': depths,

        'n_estimators': n_estimators,

        'random_state': [289]

    }),

    (LogisticRegression(), {

        'solver': ['lbfgs'],

        'C': [.05, .075, .1, .2, .35, .5, .75, 1, 2],

        'max_iter': [100, 200, 400, 600, 800, 1000],

        'random_state': [289]

    }),

    (ExtraTreesClassifier(), {

        'n_estimators': n_estimators,

        'max_depth': depths,

        'random_state': [289]

    })

]



for m_p in models:

    model, params = m_p

    params, mean, std, table = perform_search(model, params, extended=True)

    print('%s obtained %.3f +-(%.3f), best params:' % (model.__class__.__name__, mean, std), params)

    model.set_params(**params)

    model.fit(x_train, y_train)



models = [(i[0].__class__.__name__, i[0]) for i in models]

models
from sklearn.ensemble import VotingClassifier



model = VotingClassifier(models, voting='soft')

model.fit(x_train, y_train)

print('soft ensemble acc: %.3f' % acc(y_test, model.predict(x_test)))
res = pd.DataFrame()

res['PassengerId'] = test_ids

res['Survived'] = model.predict(test).astype('uint8')

res.to_csv('submission.csv', index=False)

res.head(12)