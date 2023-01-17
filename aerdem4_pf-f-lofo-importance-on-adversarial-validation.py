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

    

drop_features = ['AdoptionSpeed', 'Description', 'PetID', 'Name', 'RescuerID']

all_df.drop(columns=drop_features, inplace=True)
all_df.head(10)
!pip install lofo-importance==0.2.0
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from lightgbm import LGBMClassifier

from lofo import LOFOImportance, FLOFOImportance, plot_importance



params = {

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'metric': 'auc',

    'num_leaves': 16,

    'max_depth': -1,

    'learning_rate': 0.01,

    'verbosity': -1,

    'n_jobs': -1,

    'num_round': 500

}

features = [col for col in all_df.columns if col != "is_train"]



avd_train_df, adv_val_df = train_test_split(all_df, test_size=0.3, shuffle=True, stratify=all_df["is_train"], random_state=0)



model = LGBMClassifier(**params)

model.fit(avd_train_df[features], avd_train_df["is_train"])



# Fast version of LOFO working with already trained models

flofo_imp = FLOFOImportance(model, adv_val_df, features, 'is_train', scoring='roc_auc')

importance_df = flofo_imp.get_importance()

plot_importance(importance_df, figsize=(12, 12))
# Slower version of LOFO due to re-training, but gives more realistic feature importances

lofo_imp = LOFOImportance(all_df, features=features, target="is_train", 

                          cv=StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True), scoring="roc_auc",

                          model=LGBMClassifier(**params))

importance_df = lofo_imp.get_importance()

plot_importance(importance_df, figsize=(12, 12))