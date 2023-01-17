# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import random

random.seed(42)
train_data = pd.read_csv('../input/train/train.csv', nrows=1900000)
train_data = train_data.rename(index=str, columns={"Unnamed: 0": "id"})

train_data.head(20)
def get_info(data):

    info = pd.DataFrame(data.count() / data['target'].size, columns=['not_null'])

    info['type'] = data.dtypes

    info['unique'] = data.nunique()

    mapping = { 'feature_40': data['feature_40'].unique(),

                'feature_38': data['feature_38'].unique(),

                'feature_30': data['feature_30'].unique(),

                'feature_11': data['feature_11'].unique(),

                'feature_4': data['feature_4'].unique()}

    info['values'] = info.index.map(mapping)

    return info

    

get_info(train_data)
bad_factors = ['feature_13', 'feature_16', 'feature_32', 'feature_34', 'feature_35']

train_data = train_data.drop(bad_factors, axis=1)
numeric = train_data.select_dtypes(include=[np.number]).drop(['id','group_id','target'], axis=1).columns.tolist()
train_data = pd.get_dummies(train_data)
train_means = train_data.min(axis=0, numeric_only=True)
train_means = train_means - train_means * 2 - 999
train_data = train_data.fillna(train_means)
grouped = train_data.groupby(by='group_id', axis=0, sort=False)
for col in numeric:

    train_data[col + '_max'] = grouped[col].transform(max)

    train_data[col + '_min'] = grouped[col].transform(min)

    train_data[col + '_p_max'] = train_data[col] / (train_data[col + '_max'] + 0.001)

    train_data[col + '_p_min'] = train_data[col] / (train_data[col + '_min'] + 0.001)
train_data.head(20)
import xgboost as xgb
# import numpy as np

# from catboost import CatBoostClassifier, Pool

# limit = int(train_data.id.size * 2 / 3)

# tr = train_data.iloc[:limit, :]

# tv = train_data.iloc[limit:, :]
num_round = 30000

def model(xtr, xtv):

       ## fixed parameters

    scale_pos_weight = sum(train_data['target']==0)/sum(train_data['target']==1)  

    param = {'verbosity': 1,

             'eta': 0.025,

              "min_eta"               : 0.001,

               "eta_decay"             : 0.3,

                "max_fails"             : 3,

                "early_stopping_rounds" : 20,

             'subsample': 0.7,

             'tree_method': 'gpu_hist',

             'max_depth': 4,

             'min_child_weight':1, ## unbalanced dataset

             'objective':'binary:logistic',

             'eval_metric':'map@1', 

             'scale_pos_weight':scale_pos_weight}

    evallist = [(xtv, 'eval'), (xtr, 'train')] 

    return xgb.train(param, xtr, num_round, evallist)

    

    
def my_func(data):

    val = data['group_id'].values

    res=[]

    last = val[0]

    cur = 0

    for t in val:

        if t == last:

            cur = cur + 1

        else:

            res.append(cur)

            cur = 1

            last = t

    res.append(cur)

    return res
def to_xgb_matrix(p_data):  

    label = p_data['target']

    counts = my_func(p_data)

    data = p_data.drop(['id', 'group_id', 'target'], axis=1)

    dtrain = xgb.DMatrix(data, label=label)

    dtrain.set_group(counts)

    return dtrain



def best(data, target):

    length = len(target)

    index = 0

    rid_map = {}

    while index < length:

        tmp = data.group_id.iloc[index]

        if tmp in rid_map:

            if rid_map[tmp][0] < target[index]:

                rid_map[tmp] = [target[index], data.id.iloc[index]]

        else:

            rid_map[tmp] = [target[index], data.id.iloc[index]]

        index = index + 1

    return dict([(k, v[1]) for (k, v) in rid_map.items()])



def validate(x_tr, test):

    limit2 = int(x_tr.id.size / 2)

    tr2 = x_tr.iloc[:limit2, :]

    tv2 = x_tr.iloc[limit2:, :]



    xtr2 = to_xgb_matrix(tr2)

    xtv2 = to_xgb_matrix(tv2)



    m = model(xtr2, xtv2)

    ty = test['target']

    dd = xgb.DMatrix(test.drop(['id', 'group_id', 'target'], axis=1))



    pred = m.predict(dd, ntree_limit=m.best_ntree_limit)

    pred_val = best(test.drop(['target'], axis=1), pred)



    real_val = best(test.drop(['target'], axis=1), list(ty))

    duplicates = dict(real_val.items() & pred_val.items())



    return len(duplicates) / len(real_val)

               

#validate(tr, tv)
test_data = pd.read_csv('../input/test/test.csv')

test_data = test_data.rename(index=str, columns={"Unnamed: 0": "id"})
test_data = test_data.drop(bad_factors, axis=1)

test_data = pd.get_dummies(test_data)

test_data = test_data.fillna(train_means)

grouped2 = test_data.groupby(by='group_id', axis=0, sort=False)

for col in numeric:

    test_data[col + '_max'] = grouped2[col].transform(max)

    test_data[col + '_min'] = grouped2[col].transform(min)

    test_data[col + '_p_max'] = test_data[col] / (test_data[col + '_max'] + 0.001)

    test_data[col + '_p_min'] = test_data[col] / (test_data[col + '_min'] + 0.001)
train_data['feature_39_applicableByName'] = 0

test_data['feature_37_castVariable'] = 0

test_data['feature_37_qualifiedWithField'] = 0

test_data['feature_37_accessibleFieldGetter'] = 0

test_data = test_data.reindex(sorted(test_data.columns), axis=1)

train_data = train_data.reindex(sorted(train_data.columns), axis=1)

test_data.head(20)
limit = int(train_data.id.size * 4 / 5)

tr = train_data.iloc[:limit, :]

tv = train_data.iloc[limit:, :]
xtr2 = to_xgb_matrix(tr)

xtv2 = to_xgb_matrix(tv)

m = model(xtr2, xtv2)
tdd = xgb.DMatrix(test_data.drop(['id', 'group_id'], axis=1))

pred = m.predict(tdd, ntree_limit=m.best_ntree_limit)
mysubmission = pd.DataFrame({'Id': test_data.id, 'target': pred})

mysubmission.to_csv('submission.csv', index=False)