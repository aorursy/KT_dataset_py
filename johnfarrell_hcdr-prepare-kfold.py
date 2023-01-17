%matplotlib inline
import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
DATA_DIR = '../input/'
target_col = 'TARGET'
train = pd.read_csv(DATA_DIR+'application_train.csv')
test = pd.read_csv(DATA_DIR+'application_test.csv')
y = train[target_col].values
del train[target_col]; gc.collect()
train_num = len(train)
test_num = len(test)
from sklearn.model_selection import KFold
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=233)
train['eval_set'] = -1
for fold_i, (_, test_index) in enumerate(kf.split(np.arange(train_num))):
    train.loc[test_index, 'eval_set'] = fold_i
eval_sets = train['eval_set'].values
y.mean()
for i in range(n_splits):
    print(y[eval_sets==i].mean())
np.save('target.npy', y)
np.save('eval_sets.npy', eval_sets)