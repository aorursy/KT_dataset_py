%matplotlib inline
import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_DIR = '../input/'
target_col = 'deal_probability'
os.listdir(DATA_DIR)
usecols = ['region', 'city', 'parent_category_name', 'category_name', 
           'param_1', 'param_2', 'param_3', 'title', 'description']
train = pd.read_csv(DATA_DIR+'train.csv', usecols=usecols+[target_col])
test = pd.read_csv(DATA_DIR+'test.csv', usecols=usecols)
train.head()
train['description'].isnull().sum()
train['description'].fillna('unknown', inplace=True)
test['description'].fillna('unknown', inplace=True)
train['description'].isnull().sum()
train = train.fillna('')
test = test.fillna('')
y = train[target_col].values
del train[target_col]; gc.collect()
train_num = len(train)
df = pd.concat([train, test], ignore_index=True)
del train, test; gc.collect()

raw_cols = df.columns.tolist()
df['context'] = ''
for c in ['parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 'title']:
    df[c] = df[c].str.lower()
    df['context'] += ' ' + df[c]
df['context'].fillna('unknown', inplace=True)
df['text'] = df['description'].str.lower()
for c in raw_cols:
    del df[c]
gc.collect()
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, shuffle=True, random_state=233)

df['eval_set'] = 10 #for test
for fold_i, (_, test_index) in enumerate(kf.split(y)):
    df.loc[test_index, 'eval_set'] = fold_i
df['label'] = 2
df.loc[np.arange(train_num), 'label'] = y
df.head(10)
df.to_csv('textdata.csv', index=False)