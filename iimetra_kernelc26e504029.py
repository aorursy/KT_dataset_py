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
train_data = pd.read_csv('../input/train/train.csv')
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

get_info(train_data)

train_data = pd.get_dummies(train_data)

train_data.head(20)
train_means = train_data.mean(axis=0, numeric_only=True)
train_means
train_data = train_data.fillna(train_means)

train_data.head(20)
import xgboost as xgb

limit = int(train_data.id.size / 2)

tr = train_data.iloc[:limit, :]

tv = train_data.iloc[limit:, :]

def model(xtr, xtv):

    ## fixed parameters

    scale_pos_weight = sum(train_data['target']==0)/sum(train_data['target']==1)  

    num_rounds=10 # number of boosting iterations



    param = {'silent':1,

             'min_child_weight':1, ## unbalanced dataset

             'objective':'binary:logistic',

             'eval_metric':'auc', 

             'scale_pos_weight':scale_pos_weight}



    evallist = [(xtv, 'eval'), (xtr, 'train')] 

    num_round = 40

    return xgb.train(param, xtr, num_round, evallist, early_stopping_rounds=10)

    

    
def to_xgb_matrix(p_data):  

    label = p_data['target']

    data = p_data.drop(['id', 'group_id', 'target'], axis=1)

    dtrain = xgb.DMatrix(data, label=label)

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

               

validate(tr, tv)

    
def model(xtr, xtv):

    ## fixed parameters

    scale_pos_weight = sum(train_data['target']==0)/sum(train_data['target']==1)  

    num_rounds=10 # number of boosting iterations



    param = {'silent':1,

             'max_depth': 2,

             'min_child_weight':1, ## unbalanced dataset

             'objective':'binary:logistic',

             'eval_metric':'auc', 

             'scale_pos_weight':scale_pos_weight}



    evallist = [(xtv, 'eval'), (xtr, 'train')] 

    num_round = 150

    return xgb.train(param, xtr, num_round, evallist, early_stopping_rounds=10)

    

    
#validate(tr, tv)

0.7291788999804267
def model(xtr, xtv):

    ## fixed parameters

    scale_pos_weight = sum(train_data['target']==0)/sum(train_data['target']==1)  

    num_rounds=10 # number of boosting iterations



    param = {'silent':1,

             'max_depth': 8,

             'min_child_weight':1, ## unbalanced dataset

             'objective':'binary:logistic',

             'eval_metric':'auc', 

             'scale_pos_weight':scale_pos_weight}



    evallist = [(xtv, 'eval'), (xtr, 'train')] 

    num_round = 40

    return xgb.train(param, xtr, num_round, evallist, early_stopping_rounds=10)

    

    
#validate(tr, tv)

0.7444460755529457

def model(xtr, xtv):

    ## fixed parameters

    scale_pos_weight = sum(train_data['target']==0)/sum(train_data['target']==1)  

    num_rounds=10 # number of boosting iterations



    param = {'silent':1,

             'max_depth': 9,

             'min_child_weight':1, ## unbalanced dataset

             'objective':'binary:logistic',

             'eval_metric':'auc', 

             'scale_pos_weight':scale_pos_weight}



    evallist = [(xtv, 'eval'), (xtr, 'train')] 

    num_round = 55

    return xgb.train(param, xtr, num_round, evallist, early_stopping_rounds=10)

#validate(tr, tv)
0.7470762380113525
test_data = pd.read_csv('../input/test/test.csv')

test_data = test_data.rename(index=str, columns={"Unnamed: 0": "id"})

test_data = test_data.drop(bad_factors, axis=1)

test_data = pd.get_dummies(test_data)

test_data = test_data.fillna(train_means)

test_data.head(20)
l = ['feature_4', 'feature_10', 'feature_14', 'feature_15', 'feature_17', 'feature_18', 'feature_19', 'feature_20', 'feature_21', 'feature_22', 'feature_23', 'feature_24', 'feature_25', 'feature_26', 'feature_27', 'feature_28', 'feature_29', 'feature_30', 'feature_31', 'feature_0_False', 'feature_0_True', 'feature_1_False', 'feature_1_True', 'feature_2_False', 'feature_2_True', 'feature_3_False', 'feature_3_True', 'feature_5_False', 'feature_5_True', 'feature_6_False', 'feature_6_True', 'feature_7_False', 'feature_7_True', 'feature_8_False', 'feature_8_True', 'feature_9_False', 'feature_9_True', 'feature_11_False', 'feature_11_True', 'feature_12_False', 'feature_12_True', 'feature_33_DEPRECATED', 'feature_33_INACCESSIBLE', 'feature_33_NORMAL', 'feature_36_expected', 'feature_36_maybeExpected', 'feature_36_normal', 'feature_36_ofDefaultType', 'feature_36_unexpected', 'feature_37_accessibleFieldGetter', 'feature_37_annoMethod', 'feature_37_castVariable', 'feature_37_classNameOrGlobalStatic', 'feature_37_collectionFactory', 'feature_37_expectedTypeArgument', 'feature_37_expectedTypeConstant', 'feature_37_expectedTypeMethod', 'feature_37_expectedTypeVariable', 'feature_37_getter', 'feature_37_improbableKeyword', 'feature_37_lambda', 'feature_37_methodRef', 'feature_37_normal', 'feature_37_probableKeyword', 'feature_37_qualifiedWithField', 'feature_37_qualifiedWithGetter', 'feature_37_suitableClass', 'feature_37_superMethodParameters', 'feature_37_unlikelyClass', 'feature_37_variable', 'feature_37_verySuitableClass', 'feature_38_False', 'feature_38_True', 'feature_39_applicableByKind', 'feature_39_inapplicable', 'feature_39_unknown', 'feature_40_delegation', 'feature_40_normal', 'feature_40_recursive']



s = ['feature_4', 'feature_10', 'feature_14', 'feature_15', 'feature_17', 'feature_18', 'feature_19', 'feature_20', 'feature_21', 'feature_22', 'feature_23', 'feature_24', 'feature_25', 'feature_26', 'feature_27', 'feature_28', 'feature_29', 'feature_30', 'feature_31', 'feature_0_False', 'feature_0_True', 'feature_1_False', 'feature_1_True', 'feature_2_False', 'feature_2_True', 'feature_3_False', 'feature_3_True', 'feature_5_False', 'feature_5_True', 'feature_6_False', 'feature_6_True', 'feature_7_False', 'feature_7_True', 'feature_8_False', 'feature_8_True', 'feature_9_False', 'feature_9_True', 'feature_11_False', 'feature_11_True', 'feature_12_False', 'feature_12_True', 'feature_33_DEPRECATED', 'feature_33_INACCESSIBLE', 'feature_33_NORMAL', 'feature_36_expected', 'feature_36_maybeExpected', 'feature_36_normal', 'feature_36_ofDefaultType', 'feature_36_unexpected', 'feature_37_annoMethod', 'feature_37_classNameOrGlobalStatic', 'feature_37_collectionFactory', 'feature_37_expectedTypeArgument', 'feature_37_expectedTypeConstant', 'feature_37_expectedTypeMethod', 'feature_37_expectedTypeVariable', 'feature_37_getter', 'feature_37_improbableKeyword', 'feature_37_lambda', 'feature_37_methodRef', 'feature_37_normal', 'feature_37_probableKeyword', 'feature_37_qualifiedWithGetter', 'feature_37_suitableClass', 'feature_37_superMethodParameters', 'feature_37_unlikelyClass', 'feature_37_variable', 'feature_37_verySuitableClass', 'feature_38_False', 'feature_38_True', 'feature_39_applicableByKind', 'feature_39_applicableByName', 'feature_39_inapplicable', 'feature_39_unknown', 

'feature_40_delegation', 'feature_40_normal', 'feature_40_recursive']



print (list(set(l) - set(s)))

print(list(set(s) - set(l)))
import csv

def print_csv(prediction):

    with open('result.csv', 'w') as csv_file:

        csv_writer = csv.writer(csv_file, delimiter=',')

        csv_writer.writerow(['Id', 'target'])

        length = len(prediction)

        for i in range(0, length):

            csv_writer.writerow([i, prediction[i]])
train_data['feature_39_applicableByName'] = 0

test_data['feature_37_castVariable'] = 0

test_data['feature_37_qualifiedWithField'] = 0

test_data['feature_37_accessibleFieldGetter'] = 0

test_data = test_data.reindex(sorted(test_data.columns), axis=1)

train_data = train_data.reindex(sorted(train_data.columns), axis=1)

test_data



test_limit = int(train_data.id.size * 4 / 5)

ttr = train_data.iloc[:test_limit, :]

ttv = train_data.iloc[test_limit:, :]

mtr = to_xgb_matrix(ttr)

mtv = to_xgb_matrix(ttv)



m = model(mtr, mtv)

tdd = xgb.DMatrix(test_data.drop(['id', 'group_id'], axis=1))



pred = m.predict(tdd, ntree_limit=m.best_ntree_limit)
mysubmission = pd.DataFrame({'Id': test_data.id, 'target': pred})

mysubmission.to_csv('submission.csv', index=False)