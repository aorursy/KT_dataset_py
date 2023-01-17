# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import matplotlib as mpl 

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import cluster, preprocessing, linear_model, tree, model_selection, feature_selection

from sklearn import base, ensemble, decomposition, metrics, pipeline, datasets, impute

from skopt import gp_minimize, space, gbrt_minimize, dummy_minimize, forest_minimize

from functools import partial

import os

import lightgbm as lgb

import xgboost as xgb

import catboost as cb

from sklearn import ensemble, preprocessing, tree, model_selection, feature_selection, pipeline, metrics, svm

from imblearn import under_sampling, over_sampling, combine

from imblearn import pipeline as imb_pipeline

from imblearn import ensemble as imb_ensemble

from sklearn.model_selection import StratifiedKFold



!pip install rfpimp

train = pd.read_csv('/kaggle/input/janatahack-customer-segmentation/Train.csv')

test = pd.read_csv('/kaggle/input/janatahack-customer-segmentation/Test.csv')
# View target balance/Imbalance

train['Segmentation'].value_counts()
mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

rev_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
train.head(2)
## Label Encode and preprocess

train_copy  = train.copy()

test_copy = test.copy()

train_copy['tr'] = 1

test_copy['tr'] = 0



appended = pd.concat([train_copy, test_copy], axis = 0)



cat_cols = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1']

label_enc = {}

for col in cat_cols:

    appended[col] = appended[col].astype(str)

    enc = preprocessing.LabelEncoder().fit(appended[col])

    appended[col] = enc.transform(appended[col])

    label_enc[col] = enc



cats = ['Gender', 'Ever_Married','Graduated','Profession','Spending_Score',

'Var_1']

appended = pd.get_dummies(appended, columns = cats)

##################### create features from ID column ##############

def id_features(data):

    df = data.copy()

    df['week'] = df['ID']%7

    df['month'] = df['ID']%30

    df['year'] = df['ID']%365

    df['num_weeks'] = df['ID']//7

    df['num_year'] = df['ID']//365

    df['num_quarter'] = df['ID']//90

    df['quarter'] = df['ID']%90

    df['num_days'] = df['ID'].values - 458982

    df['num_weeks_2'] = (df['ID'].values - 458982)//7

    df['num_months_2'] = (df['ID'].values - 458982)//30



    return df

def id_features(data):

    df = data.copy()

    df['week'] = df['ID']%7

    df['month'] = df['ID']%30

    df['year'] = df['ID']%365

    df['quarter'] = df['ID']%90





    return df

appended = id_features(appended)

#appended = pd.get_dummies(appended, columns = cat_cols)

train_copy = appended.loc[appended['tr'] == 1]

test_copy = appended.loc[appended['tr'] == 0]

Xcols = appended.drop(['Segmentation', 'tr'], axis = 1).columns

'''Xcols = ['Gender', 'Ever_Married', 'Age', 'Graduated', 'Profession',

       'Work_Experience', 'Spending_Score', 'Family_Size', 'Var_1']'''

ycol = 'Segmentation'



X = train_copy[Xcols]

y = train_copy[ycol]



Xtest = test_copy[Xcols]
############ Tune Random Forest

def optimize_sk(params, param_names, X, y, scoring, estimator, cv = model_selection.StratifiedKFold(n_splits = 5)):

    '''params: list of param values

    param_names: param names

    x: training exogs

    y: training endogs

    return: negative metric after k fold validation'''



    params = dict(zip(param_names, params))



    # Initialize the model

    model = estimator(**params)



    kf = cv



    scores = []

    for train_index, test_index in kf.split(X, y):

        # Split Data

        X_train, y_train = np.array(X)[train_index, :], y[train_index]

        X_test, y_test = np.array(X)[test_index, :], y[test_index]



        # Fit model

        im = impute.KNNImputer().fit(X_train)

        X_train = im.transform(X_train)

        model.fit(X_train, y_train)



        # Evaluate model

        preds = model.predict(im.transform(X_test))

        scores.append(scoring(y_test, preds))



    return -np.mean(scores)



# Scoring

def f1_score(y_true, y_pred):

    return metrics.f1_score(y_true, y_pred, average = 'macro')



def accuracy(y_true, y_pred):

    return metrics.accuracy_score(y_true, y_pred)



# Parameter Space

param_space = [

    space.Integer(100, 1000, name = 'n_estimators'),

    space.Integer(2, 25, name = 'max_depth'),

    space.Real(0, 1, name = 'max_features'),

    space.Integer(2, 25, name = 'min_samples_leaf'),

    space.Categorical(['gini', 'entropy'], name = 'criterion'),

    space.Categorical([None, 'balanced', 'balanced_subsample'], name = 'class_weight'),

    space.Categorical([True, False], name = 'bootstrap')

]



# Param names

names = ['n_estimators', 'max_depth', 'max_features', 'min_samples_leaf', 'criterion', 'class_weight', 'bootstrap']



cat_cols =  ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1', 'ID']

cat_cols =  ['Var_1']



# Define objective - reformat it in terms of what is required for skopt

objective_optimization = partial(optimize_sk, param_names = names, X = X, y = y, 

                                scoring = accuracy, estimator = partial(ensemble.RandomForestClassifier, n_jobs = -1, random_state = 0))



# Perform Optimization

#gbrt_minimize, dummy_minimize, forest_minimize

'''skopt_optimization = gp_minimize(func = objective_optimization, 

                                dimensions = param_space, n_calls = 10, n_random_starts = 10, 

                                x0 = None, y0 = None, random_state = 10, 

                                verbose = 10)'''

skopt_optimization = dummy_minimize(func = objective_optimization, 

                                dimensions = param_space, n_calls = 10,

                                x0 = None, y0 = None, random_state = 10, 

                                verbose = 10)
model = pipeline.make_pipeline(impute.KNNImputer(), 

                               ensemble.RandomForestClassifier(**dict(zip(names, skopt_optimization.x)), 

                                                               n_jobs = -1, random_state = 0)).fit(X, y)
dict(zip(names, skopt_optimization.x))
from sklearn import impute, pipeline
model_cb = cb.CatBoostClassifier( verbose = False)



model_lgb = lgb.LGBMClassifier(n_estimators = 1000, min_samples_in_leaf = 10, learning_rate = .02, 

                          feature_fraction = .8, max_depth = 8)



# Soft Voting Classifier

model_voting = ensemble.VotingClassifier([('catboost', model_cb), ('lightgbm', model_lgb)], 

                                         voting = 'soft').fit(X, y)
# .94

model4 = pipeline.make_pipeline(impute.KNNImputer(n_neighbors = 10), ensemble.RandomForestClassifier(class_weight = 'balanced_subsample',

                    n_estimators = 200, max_depth = 20, criterion = 'entropy', max_features = .8, oob_score = True, random_state = 0)).fit(X, y)
# .94

model2 = lgb.LGBMClassifier(n_estimators=300, max_features = .85, max_depth = 15, learning_rate = 1.1).fit(X, y)
###### View Feature importance  -  Using Permutation Importance

import rfpimp



imp = rfpimp.importances(model2, X, y)

imp
pred = pd.DataFrame()

pred['ID'] = test['ID'].values

pred['Segmentation'] = pd.Series(model2.predict(Xtest))

pred.to_csv('Seg.csv', index = None)