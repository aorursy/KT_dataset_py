import pandas as pd

import numpy as np 

import os

import string



import warnings

warnings.filterwarnings("ignore")



train = pd.read_csv('../input/cat-in-the-dat/train.csv')

test = pd.read_csv('../input/cat-in-the-dat/test.csv')



print(train.shape)

print(test.shape)
Q1 = train.quantile(0.25)

Q3 = train.quantile(0.75)

IQR = Q3 - Q1

train  = train[~((train < (Q1 - 1.5 * IQR)) |(train > (Q3 + 1.5 * IQR))).any(axis=1)]
target = train['target']

train_id = train['id']

test_id = test['id']

train.drop(['target', 'id'], axis=1, inplace=True)

test.drop('id', axis=1, inplace=True)



print(train.shape)

print(test.shape)
def description(df):

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.iloc[0].values

    summary['Second Value'] = df.iloc[1].values

    summary['Third Value'] = df.iloc[2].values

    return summary

description(train)
# dictionary to map the feature

bin_dict = {'T':1, 'F':0, 'Y':1, 'N':0}



# Maping the category values in our dict

train['bin_3'] = train['bin_3'].map(bin_dict)

train['bin_4'] = train['bin_4'].map(bin_dict)

test['bin_3'] = test['bin_3'].map(bin_dict)

test['bin_4'] = test['bin_4'].map(bin_dict)
dummies = pd.concat([train, test], axis=0, sort=False)

print(f'Shape before dummy transformation: {dummies.shape}')

dummies = pd.get_dummies(dummies, columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'],\

                          prefix=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'], drop_first=True)

print(f'Shape after dummy transformation: {dummies.shape}')

train, test = dummies.iloc[:train.shape[0], :], dummies.iloc[train.shape[0]:, :]

del dummies

print(train.shape)

print(test.shape)
# Importing categorical options of pandas

from pandas.api.types import CategoricalDtype 



# seting the orders of our ordinal features

ord_1 = CategoricalDtype(categories=['Novice', 'Contributor','Expert', 

                                     'Master', 'Grandmaster'], ordered=True)

ord_2 = CategoricalDtype(categories=['Freezing', 'Cold', 'Warm', 'Hot',

                                     'Boiling Hot', 'Lava Hot'], ordered=True)

ord_3 = CategoricalDtype(categories=['a', 'b', 'c', 'd', 'e', 'f', 'g',

                                     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'], ordered=True)

ord_4 = CategoricalDtype(categories=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',

                                     'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',

                                     'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], ordered=True)

def transformingOrdinalFeatures(df, ord_1, ord_2, ord_3, ord_4):    

    df.ord_1 = df.ord_1.astype(ord_1)

    df.ord_2 = df.ord_2.astype(ord_2)

    df.ord_3 = df.ord_3.astype(ord_3)

    df.ord_4 = df.ord_4.astype(ord_4)



    # Geting the codes of ordinal categoy's 

    df.ord_1 = df.ord_1.cat.codes

    df.ord_2 = df.ord_2.cat.codes

    df.ord_3 = df.ord_3.cat.codes

    df.ord_4 = df.ord_4.cat.codes

    

    return df

train = transformingOrdinalFeatures(train, ord_1, ord_2, ord_3, ord_4)

test = transformingOrdinalFeatures(test, ord_1, ord_2, ord_3, ord_4)



print(train.shape)

print(test.shape)
def date_cyc_enc(df, col, max_vals):

    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_vals)

    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_vals)

    return df



train = date_cyc_enc(train, 'day', 7)

test = date_cyc_enc(test, 'day', 7) 



train = date_cyc_enc(train, 'month', 12)

test = date_cyc_enc(test, 'month', 12)



print(train.shape)

print(test.shape)
import string



# Then encode 'ord_5' using ACSII values



# Option 1: Add up the indices of two letters in string.ascii_letters

train['ord_5_oe_add'] = train['ord_5'].apply(lambda x:sum([(string.ascii_letters.find(letter)+1) for letter in x]))

test['ord_5_oe_add'] = test['ord_5'].apply(lambda x:sum([(string.ascii_letters.find(letter)+1) for letter in x]))



# Option 2: Join the indices of two letters in string.ascii_letters

train['ord_5_oe_join'] = train['ord_5'].apply(lambda x:float(''.join(str(string.ascii_letters.find(letter)+1) for letter in x)))

test['ord_5_oe_join'] = test['ord_5'].apply(lambda x:float(''.join(str(string.ascii_letters.find(letter)+1) for letter in x)))



# Option 3: Split 'ord_5' into two new columns using the indices of two letters in string.ascii_letters, separately

train['ord_5_oe1'] = train['ord_5'].apply(lambda x:(string.ascii_letters.find(x[0])+1))

test['ord_5_oe1'] = test['ord_5'].apply(lambda x:(string.ascii_letters.find(x[0])+1))



train['ord_5_oe2'] = train['ord_5'].apply(lambda x:(string.ascii_letters.find(x[1])+1))

test['ord_5_oe2'] = test['ord_5'].apply(lambda x:(string.ascii_letters.find(x[1])+1))



for col in ['ord_5_oe1', 'ord_5_oe2', 'ord_5_oe_add', 'ord_5_oe_join']:

    train[col]= train[col].astype('float64')

    test[col]= test[col].astype('float64')



print(train.shape)

print(test.shape)
from sklearn.preprocessing import LabelEncoder



high_card_feats = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']



for col in high_card_feats:    

    train[f'hash_{col}'] = train[col].apply( lambda x: hash(str(x)) % 5000 )

    test[f'hash_{col}'] = test[col].apply( lambda x: hash(str(x)) % 5000 )

                                               

    if train[col].dtype == 'object' or test[col].dtype == 'object': 

        lbl = LabelEncoder()

        lbl.fit(list(train[col].values) + list(test[col].values))

        train[f'le_{col}'] = lbl.transform(list(train[col].values))

        test[f'le_{col}'] = lbl.transform(list(test[col].values))



print(train.shape)

print(test.shape)
col = high_card_feats + ['ord_5','day', 'month']

                    # + ['hash_nom_6', 'hash_nom_7', 'hash_nom_8', 'hash_nom_9']

                    # + ['le_nom_5', 'le_nom_6', 'le_nom_7', 'le_nom_8', 'le_nom_9']

train.drop(col, axis=1, inplace=True)

test.drop(col, axis=1, inplace=True)



print(train.shape)

print(test.shape)
def get_optimized_column(column):

    if not np.issubdtype(column.dtypes, np.number):

        return column

    integers = [np.int8, np.int16, np.int32, np.int64]

    floats = [np.float16, np.float32, np.float64]

    max = column.max()

    relevant_types = integers if np.issubdtype(column.dtypes, np.integer) else floats

    for dtype in relevant_types:

        if dtype(max) == max:

            return column.astype(dtype)

    return column





def reduce_mem_usage(df, verbose=True):

    start_mem = df.memory_usage().sum() / 1024**2    

    for column in df.columns:

        df[column] = get_optimized_column(df[column]) 

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

train = reduce_mem_usage(train)

test = reduce_mem_usage(test)

description(train)
from catboost import CatBoostClassifier, Pool # https://github.com/catboost/tutorials/blob/master/python_tutorial.ipynb

from sklearn.metrics import roc_auc_score as auc

from sklearn.model_selection import train_test_split



def runCatBoost(train_pool, validate_pool, params):

    print('Train CatBoost model')    

    model = CatBoostClassifier(**params)

    model.fit(train_pool, eval_set=validate_pool, plot=True)

    return model

    

def predict(model, pool):

    print('Predict')

    pred_test_y = model.predict_proba(X_test)[:, 1]

    return pred_test_y



X_train, X_val, y_train, y_val = train_test_split(

    train, target,

    test_size = 0.3,

    random_state = 42

)

cat_features = ['bin_0','bin_1','bin_2','bin_3','bin_4','ord_0','ord_1','ord_2','ord_3','ord_4']

train_pool = Pool(train, target, cat_features = cat_features)

validate_pool = Pool(X_val, y_val, cat_features = cat_features)

test_pool = Pool(data=test, cat_features = cat_features)


params = {

    'iterations': 15000,

    'learning_rate': 0.1,

    'eval_metric': 'AUC',

    'random_seed': 42,

    'task_type':'GPU',

    'l2_leaf_reg': 10,

    'logging_level': 'Silent',

    'use_best_model': True,

    'custom_loss': ['AUC'], 

    'metric_period': 50,

    'bagging_temperature' : 0.2,

    'l2_leaf_reg': 10

    }

model1 = runCatBoost(train_pool, validate_pool, params)
params.update({

    'random_seed': 42,

    'learning_rate': 0.05,

    'l2_leaf_reg': 5,

    'depth': 4

})

#model2 = runCatBoost(train_pool, validate_pool, params)
params.update({

    'random_seed': 44,

    'learning_rate': 0.025,

    'l2_leaf_reg': 3,

    'depth': 7

})

#model3 = runCatBoost(train_pool, validate_pool, params)
params.update({

    'iterations': 15000,

    'random_seed': 45,

    'learning_rate': 0.0125,

    'l2_leaf_reg': 10,

    'depth': 5

})

#model4 = runCatBoost(train_pool, validate_pool, params)
def submit(model, pool, i):

    print('Predict ' + str(i))

    results = model.predict_proba(pool)[:, 1]

    submission = pd.DataFrame({'id': test_id, 'target': results})

    submission.to_csv('submission' + str(i) + '.csv', index=False)

submit(model1, test_pool,1)

#submit(model2, test_pool,2)

#submit(model3, test_pool,3)

#submit(model4, test_pool,4)