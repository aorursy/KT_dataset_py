import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_regression as mir

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold

from sklearn.cluster import KMeans

from sklearn import preprocessing

from sklearn.metrics import log_loss

from collections import Counter

import seaborn as sns

import lightgbm as lgbm

from sklearn import metrics

from sklearn import model_selection

from tqdm import tqdm



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)
test_features        = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

train_features       = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

sample_submission    = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')



g_feat  = [col for col in train_features if 'g-' in col] 

c_feat  = [col for col in train_features if 'c-' in col]

cp_feat = [col for col in train_features if 'cp_' in col]

targets = [col for col in sample_submission.columns[1:]]
# feat    = ['g-128',

#  'g-231',

#  'g-600',

#  'g-201',

#  'g-385',

#  'g-300',

#  'g-207',

#  'g-357',

#  'c-68',

#  'g-168',

#  'g-392',

#  'g-180',

#  'g-22',

#  'g-269',

#  'g-91',

#  'g-522',

#  'g-312',

#  'g-84',

#  'g-708',

#  'g-701',

#  'g-562',

#  'g-75',

#  'g-628',

#  'g-166',

#  'g-122',

#  'g-47',

#  'g-68',

#  'g-208',

#  'g-157',

#  'c-14',

#  'c-98',

#  'g-206',

#  'g-636',

#  'g-100',

#  'g-34',

#  'g-175',

#  'c-65',

#  'g-202',

#  'g-512',

#  'g-689'] 
feature_pick = pd.read_csv('../input/notebookf054760cbd/feature_pick.csv')
feat_d = {}

for id_, gr in feature_pick.groupby('targets'):

    feat_d[id_] = [x for x in gr['index'] if 'cp_' not in x]
top_50 = train_targets_scored[targets].sum(axis=0).sort_values(ascending=False).index[:50]
merge_df  =  pd.concat([train_features, train_targets_scored], axis=1)

merge_df  =  merge_df[merge_df['cp_type']!='ctl_vehicle']
for target in top_50:

    feat = feat_d[target]

    

    df0  = merge_df[feat+[target]]

    df0  = df0.melt(id_vars=[target], var_name='feat', value_name='value').astype({'value': 'float32'})



    plt.figure(figsize=(30, 8))

    sns.boxplot(data=df0, x='feat', y='value', hue=target)

    plt.show()