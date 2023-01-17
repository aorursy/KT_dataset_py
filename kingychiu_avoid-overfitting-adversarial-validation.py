from sklearn.model_selection import KFold, StratifiedKFold

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy as sp

from collections import Counter

from functools import partial

from math import sqrt

from sklearn.metrics import cohen_kappa_score, mean_squared_error

from sklearn.metrics import confusion_matrix as sk_cmatrix

import lightgbm as lgb

import os

from itertools import combinations

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import  mean_squared_error

import glob

import json

from tqdm import tqdm

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



random_state = 1999
train_df = pd.read_csv('../input/train/train.csv')

test_df = pd.read_csv('../input/test/test.csv')

state_df = pd.read_csv('../input/state_labels.csv')

color_df = pd.read_csv('../input/color_labels.csv')

breed_df = pd.read_csv('../input/breed_labels.csv')

sample_submission_df = pd.read_csv('../input/test/sample_submission.csv')

train_df['is_train'] = 1

test_df['is_train'] = 0



all_df = pd.concat([train_df, test_df]).reset_index(drop=True)



all_df['description_length'] = all_df['Description'].str.len()

all_df['name_length'] = all_df['Name'].str.len()



'''Label Encode'''

for f in ['RescuerID']:

    le = LabelEncoder()

    all_df[f] = le.fit_transform(all_df[f])



'''TFIDF'''

tfidf_configs = [

    ('Description', (1, 1), 3),

]

for tfidf_config in tfidf_configs:

    feature = tfidf_config[0]

    ngram_range = tfidf_config[1]

    svd_size = tfidf_config[2]

    all_df[feature] = all_df[feature].fillna('')

    vectorizer = TfidfVectorizer(analyzer='word', stop_words = 'english', ngram_range=ngram_range)

    svd = TruncatedSVD(n_components=svd_size, n_iter=100, random_state=random_state)

    vector = vectorizer.fit_transform(all_df[feature])

    vector = svd.fit_transform(vector)

    vector_df = pd.DataFrame(vector)

    vector_df = vector_df.add_prefix('TFIDF_{}_'.format(feature))

    all_df = pd.concat([all_df, vector_df], axis=1)

    

drop_features = ['AdoptionSpeed', 'Description', 'PetID', 'Name']

all_df.drop(columns=drop_features, inplace=True)
all_df.head(10)
params = {

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'metric': 'auc',

    'num_leaves': 16,

    'max_depth': -1,

    'learning_rate': 0.01,

    'verbosity': -1,

    'n_jobs': -1,

}

importance = []

kf = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=False)

for train_index, test_index in kf.split(all_df, all_df['is_train']):

    train_x, valid_x = all_df.loc[train_index], all_df.loc[test_index]

    train_y = train_x.pop('is_train')

    valid_y = valid_x.pop('is_train')

    features_used = train_x.columns

    

    train_data = lgb.Dataset(train_x, label=train_y)

    valid_data = lgb.Dataset(valid_x, label=valid_y)



    clf = lgb.train(params,

            train_data,

            1000,

            valid_sets=[train_data, valid_data],

            verbose_eval=100000,

            early_stopping_rounds=100,

            )

    

    # feature importance

    importance.append(clf.feature_importance(importance_type='gain'))

    

importance = np.array(importance)

imp_sum = np.sum(importance, axis=0)

imp_mean = np.mean(importance, axis=0)

imp_std = np.std(importance, axis=0)

imp_std[imp_std == 0] = 1

importance_df = pd.DataFrame({

    'feature': features_used,

    'sum': imp_sum,

    'mean': imp_mean,

    'std': imp_std,

    'mean/std': imp_mean / imp_std

})

importance_df.sort_values('mean', ascending=False, inplace=True)

importance_df
train_df = all_df[all_df['is_train']==1]

test_df = all_df[all_df['is_train']==0]



def compare_categorical_features(f):

    train_values = list(train_df[f].value_counts().index)

    test_values = list(test_df[f].value_counts().index)

    common_values = np.intersect1d(train_values, test_values)

    print(f, '# train values', len(train_values), '# common values:', len(common_values), ' overlap ratio:',100* len(common_values)/len(train_values), '%')



def compare_numerical_features(f):

    sns.set(style="whitegrid")

    g = sns.violinplot(data=all_df, x='is_train', y=f)

    
for f in [ 'State', 'Health', 'Gender', 'Vaccinated', 'Sterilized', 'Dewormed', 'Quantity', 'Type', 'Breed1', 'Breed2', 'Color1', 'Color2', 'Color3']:

    compare_categorical_features(f)

    print('--------')
compare_numerical_features('description_length')
compare_numerical_features('TFIDF_Description_0')
compare_numerical_features('TFIDF_Description_1')
compare_numerical_features('TFIDF_Description_2')
compare_numerical_features('Age')
compare_numerical_features('name_length')