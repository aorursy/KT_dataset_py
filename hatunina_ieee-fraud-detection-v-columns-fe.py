import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm_notebook as tqdm



from sklearn import preprocessing



import os

print(os.listdir("../input"))

import gc



pd.set_option('display.max_columns', 500)

%matplotlib inline
train = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')



print('train shape: {}'.format(train.shape))

print('train_transaction shape: {}'.format(train_transaction.shape))
def search_min_multi_vals(vals):

    pass_vals = []

    min_multi_vals = []

    for val in tqdm(vals):

        tmp_vals_1 = vals[np.where(vals%val == 0)]

        if len(tmp_vals_1) > 2:

            tmp_vals_2 = [val / tmp_val for tmp_val in val/vals if tmp_val.is_integer()]

            min_val = min(tmp_vals_2)

            if min_val not in min_multi_vals:

                min_multi_vals.append(min_val)

        else:

            pass_vals.append(val)

    print('min_multi_vals: {}'.format(len(min_multi_vals)))

    print('pass_vals: {}'.format(len(pass_vals)))

    return min_multi_vals, pass_vals
def adjust_vals(min_multi_vals, pass_vals):

    rm_vals = []

    pass_array = np.array(pass_vals)

    for val in min_multi_vals:

        rm_vals.extend(list(pass_array[np.where(pass_array%val==0)]))



    u_rm_vals = set(rm_vals)



    add_multi_vals = list(set(pass_vals).difference(u_rm_vals))



    assert len(set(min_multi_vals).intersection(set(add_multi_vals))) == 0, print('something error, please check process.')

    

    multi_vals = []

    multi_vals.extend(min_multi_vals)

    multi_vals.extend(add_multi_vals)

    

    print('multi_vals: {}'.format(len(multi_vals)))

    

    return multi_vals
def category_mapping(val_category_dict, raw_array):    

    category_array = np.zeros(len(raw_array))

    for val, category in tqdm(val_category_dict.items()):

        category_array[np.where(raw_array%val==0)] = category



    # zero

    category_array[np.where(raw_array == 0)] = -1

    # nan

    category_array[np.where(np.isnan(raw_array))] = -2

    category_array = category_array.astype('int')



    assert np.all(raw_array[category_array==-1]==0), print('something error, please check zero process.')

    assert np.all(np.isnan(raw_array[category_array==-2])), print('something error, please check nan process.')

    return category_array
def category_pipeline(v_feats, df):

    v_categories = {}

    for v_feat in v_feats:

        print('RUN: {}'.format(v_feat))

        v_vals = df[v_feat].unique()

        min_multi_vals, pass_vals = search_min_multi_vals(v_vals)

        adjusted_multi_vals = adjust_vals(min_multi_vals, pass_vals)



        val_category_dict = {val: category for val, category in zip(adjusted_multi_vals, range(len(adjusted_multi_vals)))}



        raw_array = df[v_feat].values

        v_category = category_mapping(val_category_dict, raw_array)

        v_categories[v_feat] = v_category

    return v_categories
v_feats_1 = ['V202', 'V203', 'V204']

v_categories = category_pipeline(v_feats_1, train_transaction)

for v_feat, v_category in v_categories.items():

    train_transaction['{}_category'.format(v_feat)] = v_category

    counts_dict = train_transaction[v_feat].value_counts()

    train_transaction['{}_category_cnt'.format(v_feat)] = train_transaction[v_feat].map(counts_dict)
train_transaction[['TransactionID', 'isFraud', 'TransactionDT', 'V202', 'V203', 'V204', 'V202_category', 'V203_category', 'V204_category', 'V202_category_cnt', 'V203_category_cnt', 'V204_category_cnt']].head(20)