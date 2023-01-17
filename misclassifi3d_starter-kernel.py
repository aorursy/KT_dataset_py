from collections import Counter

import datetime

import gc

from time import time





import numpy as np

import pandas as pd

import multiprocessing

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from tqdm import tqdm_notebook

# from sklearn.preprocessing import LabelEncoder

# from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit, train_test_split

# from sklearn.metrics import roc_auc_score

# from sklearn.tree import DecisionTreeClassifier

# from sklearn import tree

# import graphviz

warnings.simplefilter('ignore')

sns.set()

%matplotlib inline
# Use multiprocessing to fast reading files in

files = ['../input/test_identity.csv', 

         '../input/test_transaction.csv',

         '../input/train_identity.csv',

         '../input/train_transaction.csv',

         '../input/sample_submission.csv']



def load_data(file):

    return pd.read_csv(file)



with multiprocessing.Pool() as pool:

    test_id, test_tr, train_id, train_tr, sub = pool.map(load_data, files)
# Train and Test set

train = pd.merge(train_tr, train_id, on='TransactionID', how='left')

test = pd.merge(test_tr, test_id, on='TransactionID', how='left')
del test_id, test_tr, train_id, train_tr

gc.collect()
# Print Shapes

print("Train Dataset shape: ", train.shape)

print("Test Dataset shape: ", test.shape)
# Print Y variable distribution For Training Set 

train['isFraud'].value_counts(normalize = True)
# Print Proportions for categorical variables



for i in train.select_dtypes('object').columns:

    

    print(pd.crosstab(train['isFraud'], train[i], normalize = 'index'))

    print("-----------------------------")

    print("-----------------------------")

    
median_diffs = []



for i in train.select_dtypes('number').columns[2:]:

    

    # Normalize numerical feature

    var_m = train[i].mean()

    var_std = train[i].std()

    norm_var = (train[i] - var_m) / var_std

    

    # Create subset

    sub = train[[i, 'isFraud']]

    sub[i] = norm_var

    

    # Find absolute difference in normalized median

    temp = sub.groupby('isFraud').median().reset_index()

    abs_median_diff = abs(temp.iloc[0][i] - temp.iloc[1][i])

    

    median_diffs.append((abs_median_diff, i))
# Top 10 features by absolute normalized median difference

sorted(median_diffs, reverse = True)[0:10]
sns.boxplot(train['isFraud'], train['V50'])