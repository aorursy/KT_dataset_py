import math

import json

import configparser

# import numpy as np

import pandas as pd

from IPython.display import display

from subprocess import check_output

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
def get_base_model(modelname):

    if modelname == 'log_reg':

        return LogisticRegression()

    elif modelname == 'svc':

        return SVC()

    elif modelname == 'l_svc':

        return LinearSVC()

    elif modelname == 'rf_clf':

        return RandomForestClassifier()

    elif modelname == 'gbdt_clf':

        return GradientBoostingClassifier()

    elif modelname == 'knn_clf':

        return KNeighborsClassifier()

    elif modelname == 'g_nb':

        return GaussianNB()

    elif modelname == 'preceptron':

        return Perceptron()

    elif modelname == 'sgd_clf':

        return SGDClassifier()

    elif modelname == 'dt_clf':

        return DecisionTreeClassifier()



    

def display_dfs(dfs):

    for df in dfs:

        display(df.head())

        display(df.describe())





def replace_dfs(dfs, target, config, *, mean=None):

    """

        config is dict

    """

    output = []

    for df in dfs:

        for i, value1 in enumerate(df[target].values):

            for key, value2 in config.items():

                # mean

                if value2 == 'mean':

                    value2 = mean

                # translate

                if key == 'isnan':

                    if math.isnan(value1):

                        df[target].values[i] = value2

                else:

                    if value1 == key:

                        df[target].values[i] = value2

        output.append(df)

    return output





def categorize_dfs(dfs, target, config):

    """

        config is list

    """

    output = []

    for df in dfs:

        for value2 in config:

            df['%s_%s' % (target, value2)] = [0] * len(df[target].values)

        for i, value1 in enumerate(df[target].values):

            for value2 in config:

                if value1 == value2:

                    df['%s_%s' % (target, value2)].values[i] = 1

        del df[target]

        output.append(df)

    return output





def to_float_dfs(dfs, pred_col, id_col):

    output = []

    for df in dfs:

        for key in df.keys():

            if key == pred_col or key == id_col:

                continue

            df[key] = df[key].astype(float)

        output.append(df)

    return output





def translate_age(dfs, train_df):

    #######################################

    # no age => mean grouped Mr, Mrs, Miss

    #######################################

    # get honorific title

    train_df['HonorificTitle'] = [''] * len(train_df['Name'].values)

    for i, val in enumerate(train_df['Name'].values):

        train_df['HonorificTitle'].values[i] = val.split(',')[1].split('.')[0]

    # get honorific title => age mean

    t2m = {}

    tmp = train_df.groupby('HonorificTitle')['Age']

    for key in tmp.indices.keys():

        t2m[key] = tmp.mean()[key]

    # age range

    for df in dfs:

        for i, val in enumerate(df['Age'].values):

            if math.isnan(val):

                val = t2m[df['Name'].values[i].split(',')[1].split('.')[0]]

                df['Age'].values[i] = val

    # del honorific title

    del train_df['HonorificTitle']

    return dfs





def translate_familystatus(dfs, train_df):

    #######################################

    # sibsp + parch == 0 => no family(0)

    # survive vs no survive in same familyname

    # => more survive(1) or less survive(2)

    # categorize after

    #######################################

    # get family name

    train_df['FamilyName'] = [''] * len(train_df['Name'].values)

    for i, val in enumerate(train_df['Name'].values):

        train_df['FamilyName'].values[i] = val.split(',')[0]

    # get family name => family status

    n2s = {}

    tmp = train_df.groupby('FamilyName')['Survived']

    for key in tmp.indices.keys():

        survived_num = tmp.sum()[key]

        no_survived_num = tmp.count()[key] - survived_num

        if survived_num > no_survived_num:

            n2s[key] = 1

        else:

            n2s[key] = 2

    # main

    for df in dfs:

        df['FamilyStatus'] = [0] * len(df['Name'].values)

        for i, val in enumerate(df['Name'].values):

            family_name = val.split(',')[0]

            # no family

            if df['SibSp'].values[i] + df['Parch'].values[i] == 0:

                continue

            # any family

            if family_name in n2s:

                df['FamilyStatus'].values[i] = n2s[family_name]

    # del family name

    del train_df['FamilyName']

    return dfs
# definition

# for jupyter

config_txt = """

[data]

path = ../input/

pred_col = Survived

id_col = PassengerId

train_num =



[model]

base = gbdt_clf

scoring = accuracy

params = {

    "n_estimators": [1, 2, 3, 4, 5],

    "max_depth": [1, 2, 3, 4, 5]

    }



[translate]

replace = {

    }

adhoc = [

    "translate_age",

    "translate_familystatus"

    ]

category = {

    "Sex": ["male", "female"],

    "Pclass": [1, 2, 3],

    "FamilyStatus": [0, 1, 2]

    }

del = [

    "Name",

    "SibSp",

    "Parch",

    "Ticket",

    "Fare",

    "Cabin",

    "Embarked"

    ]

"""



cp = configparser.SafeConfigParser()

cp.read_string(config_txt)

# data

data_path = cp.get('data', 'path')

pred_col = cp.get('data', 'pred_col')

id_col = cp.get('data', 'id_col')

train_num = cp.get('data', 'train_num')

if train_num:

    train_num = int(train_num)

else:

    train_num = None

# model

base_model = get_base_model(cp.get('model', 'base'))

scoring = cp.get('model', 'scoring')

params = json.loads(cp.get('model', 'params'))

# traslate

trans_adhoc = json.loads(cp.get('translate', 'adhoc'))

trans_replace = json.loads(cp.get('translate', 'replace'))

trans_del = json.loads(cp.get('translate', 'del'))

trans_category = json.loads(cp.get('translate', 'category'))
# data list

print('### DATA LIST')

print(check_output(['ls', data_path]).decode('utf8'))

train_df = pd.read_csv('%strain.csv' % data_path)

test_df = pd.read_csv('%stest.csv' % data_path)
# init overview

print('### INIT OVERVIEW')

display_dfs([train_df, test_df])
# data translation

# replace

for key, value in trans_replace.items():

    # mean

    if train_df.dtypes[key] == 'object':

        key_mean = None

    else:

        key_mean = train_df[key].mean()

    train_df, test_df = replace_dfs(

        [train_df, test_df], key, value, mean=key_mean)

# adhoc

for value in trans_adhoc:

    train_df, test_df = eval(

        '%s' % value)([train_df, test_df], train_df)

# category

for key, values in trans_category.items():

    train_df, test_df = categorize_dfs([train_df, test_df], key, values)

# del

for value in trans_del:

    del train_df[value]

    del test_df[value]

# float

train_df, test_df = to_float_dfs([train_df, test_df], pred_col, id_col)
# translation overview

print('### TRANSLATION OVERVIEW')

display_dfs([train_df, test_df])
# simple visualization

print('### SIMPLE VISUALIZATION')

for key in train_df.keys():

    if key == pred_col or key == id_col:

        continue

    g = sns.FacetGrid(train_df, col=pred_col)

    g.map(plt.hist, key, bins=20)
# create ndarray

Y_train = train_df[pred_col].values

id_pred = test_df[id_col].values

del train_df[pred_col]

del train_df[id_col]

del test_df[id_col]

X_train = train_df.values

X_test = test_df.values
# translate ndarray to standard

ss = StandardScaler()

ss.fit(X_train)

X_train = ss.transform(X_train)

X_test = ss.transform(X_test)
# fit

print('### FIT')

gs = GridSearchCV(base_model, params, scoring=scoring, n_jobs=-1)

if train_num:

    print('train num: %s' % train_num)

    gs.fit(X_train[:train_num], Y_train[:train_num])

else:

    print('train num: ALL(%s)' % len(X_train))

    gs.fit(X_train, Y_train)

print('X train shape: %s' % str(X_train.shape))

print('Y train shape: %s' % str(Y_train.shape))

print('best params: %s' % gs.best_params_)

print('best score of trained grid search: %s' % gs.best_score_)

if train_num:

    print(

        'best score of not trained data: %s' %

        gs.score(X_train[train_num:], Y_train[train_num:]))

best_model = gs.best_estimator_

print('best model: %s' % best_model)
# predict and create output

Y_pred = best_model.predict(X_test)

print('%s,%s' % (id_col, pred_col))

for i in range(len(id_pred)):

    print('%s,%s' % (id_pred[i], Y_pred[i]))