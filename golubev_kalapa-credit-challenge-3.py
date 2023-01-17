import warnings

warnings.filterwarnings("ignore")



from ast import literal_eval

from itertools import combinations

from unidecode import unidecode



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold



import lightgbm as lgb

import numpy as np

import pandas as pd



short_columns = dict([(f'FIELD_{i}', str(i)) for i in range(1, 58)])

ignore_columns = '36 37 label id'.split()

unidecode_columns = 'province district maCv'.split()

object_columns = '7 8 9 10 11 12 13 17 18 19 20 23 24 25 26 27 28 29 3 30 31 35 38 39 40 42 43 44 45 district maCv province'.split()

oh_columns = '8 9 10 11 12 13 17 18 19 20 23 24 25 26 27 28 29 3 30 31 35 38 39 40 41 42 43 44 45 province'.split()

auto_columns = '1 2 3 4 5 6 14 15 16 21 22 32 33 34 46 50 51 52 53 54 55 56 57'.split()

elemenets_f7 = """AT BT CB CC CH CK CN DK DN DT GB GD HC HD HG HK HN HS HT HX KC LS 

                  MS ND NN NO PV QN QT SV TA TB TC TE TK TL TN TQ TS XB XD XK XN XV""".split()

params = {

    'objective'         : 'binary',    

    'metric'            : 'auc', 

    'nthread'           : 4,

    'learning_rate'     : 0.01,



    'num_leaves'        : 23,

    'feature_fraction'  : 0.106,

    'bagging_fraction'  : 0.825,

    'max_depth'         : -1,

    'lambda_l1'         : 0.2,

    'lambda_l2'         : 2.7,

    'min_split_gain'    : 0.007,

}



def gini(y_true, y_score):

    return roc_auc_score(y_true, y_score)*2 - 1



def lgb_gini(y_pred, dataset_true):

    y_true = dataset_true.get_label()

    return 'gini', gini(y_true, y_pred), True



def f3_to_year(x):

    if x == 0 or x == -1:

        return 0

    for i in range(1, 23):

        t = int(i*365.25) - 13

        if t - 30 <= x <= t + 30:

            return min(12, i)

    return -999



def transform(df):

    df.rename(columns=short_columns, inplace=True)

    columns = set(df.columns).difference(ignore_columns)

    

    for l, r in combinations(auto_columns, 2):

        for func in 'add subtract divide multiply'.split():

            df[f'auto_{func}_{l}_{r}'] = getattr(np, func)(df[l], df[r])



    df['sum_14_15_32_33_34'] = df['14 15 32 33 34'.split()].sum(axis=1)

    

    f7_array = df['7'].apply(lambda x: '[]' if x != x else x).apply(literal_eval)

    df['cnt_7'] = f7_array.apply(len)

    for col in elemenets_f7:

        df[f'cnt_7_{col}'] = f7_array.apply(lambda x: x.count(col))

        

    df[unidecode_columns] = df[unidecode_columns].applymap(lambda x: unidecode(x).lower() if x == x else x)

        

    for func in 'equal less greater subtract add'.split():

        df[f'age1_{func}_age2'] = getattr(np, func)(df['age_source1'], df['age_source2'])



    df['3'] = df['3'].apply(f3_to_year)

    df['41'] = df['41'].replace({'I':1, 'II':2, 'III':3, 'IV':4, 'V':5, 'None':np.NaN}).astype(float)

    

    df['cnt_NaN'] = df[columns].isna().sum(axis=1)

    df['cnt_True'] = df[columns].applymap(lambda x: isinstance(x, bool) and x).sum(axis=1)

    df['cnt_False'] = df[columns].applymap(lambda x: isinstance(x, bool) and not x).sum(axis=1)

    for name in 'TRUE FALSE None'.split():

        df[f'cnt_{name}'] = df[columns].applymap(lambda x: x == name).sum(axis=1)

    for l, r in combinations('TRUE FALSE True False None NaN'.split(), 2):

        df[f'cnt_{func}_{l}_{r}'] = df[f'cnt_{l}'] + df[f'cnt_{r}']

        

    df = pd.concat([df, pd.get_dummies(df[oh_columns], columns=oh_columns, dummy_na=True).add_prefix('_')], axis=1)

    df[object_columns] = df[object_columns].astype('category')

    return df



train = pd.read_csv('../input/kalapacredit/train.csv')

test  = pd.read_csv('../input/kalapacredit/test.csv')

label = train.pop('label')



train = transform(train)

test = transform(test)

columns = sorted(set(train.columns).intersection(test.columns).difference(ignore_columns))



X_train, X_test, y_train = train[columns], test[columns], label

skf = StratifiedKFold(n_splits=4, random_state=3462873, shuffle=True)

preds = 0.0

for itrain, ivalid in skf.split(X_train, y_train):

    lgb_train = lgb.Dataset(X_train.iloc[itrain], y_train.iloc[itrain])

    lgb_eval  = lgb.Dataset(X_train.iloc[ivalid], y_train.iloc[ivalid], reference = lgb_train)

    model = lgb.train(params,

                lgb_train,

                num_boost_round = 99999,  

                early_stopping_rounds = 800,

                feval = lgb_gini,

                verbose_eval = False,

                valid_sets = [lgb_train, lgb_eval])

    pred = model.predict(X_test)

    preds += pred/skf.n_splits



test['label'] = preds

test[['id', 'label']].to_csv('submission.csv', index=False)