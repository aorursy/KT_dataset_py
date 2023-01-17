# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



from lightgbm import LGBMClassifier, Dataset

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.metrics import roc_auc_score

from matplotlib import pyplot as plt



from scipy.stats import ttest_ind, ttest_rel
#функция для загрузки покупок

def load_purchases(path, cols=None, nrows=None):

    """

        path: путь до файла с покупками,

        cols: list колонок для загрузки,

        nrows: количество строк датафрейма

    """

    

    dtypes = {

        'regular_points_received': np.float16,

        'express_points_received': np.float16,

        'regular_points_spent': np.float16,

        'express_points_spent': np.float16,

        'purchase_sum': np.float32,

        'product_quantity': np.float16,

        'trn_sum_from_iss': np.float32,

        'trn_sum_from_red': np.float16,        

    }

    if cols:

        purchases = pd.read_csv(path, dtype=dtypes, parse_dates=['transaction_datetime'], nrows=nrows, usecols=cols)

    else:

        purchases = pd.read_csv(path, dtype=dtypes, parse_dates=['transaction_datetime'], nrows=nrows)

    purchases['purchase_sum'] = np.round(purchases['purchase_sum'], 2)

    

    return purchases
# загружаем данные

train_df = pd.read_csv('../input/x5-uplift-valid/data/train.csv')

test_df = pd.read_csv('../input/x5-uplift-valid/data/test.csv')



clients = pd.read_csv('../input/x5-uplift-valid/data/clients2.csv')



puchases_train = load_purchases('../input/x5-uplift-valid/train_purch/train_purch.csv')

puchases_test = load_purchases('../input/x5-uplift-valid/test_purch/test_purch.csv')



product_df = pd.read_csv('../input/x5-uplift-valid/data/products.csv')
train_df['new_target'] = 0

train_df.loc[(train_df['treatment_flg'] == 1) & (train_df['target'] == 1), 'new_target'] = 1

train_df.loc[(train_df['treatment_flg'] == 0) & (train_df['target'] == 0), 'new_target'] = 1
train_df
product_df
puchases_train
clients
popular_store = puchases_train['store_id'].value_counts().idxmax()

puchases_train['store'] = puchases_train['store_id'].apply(lambda x: int(x == popular_store))

pop_store_counter = puchases_train.groupby('client_id')['store'].sum()

train_df['store_count'] = train_df['client_id'].map(pop_store_counter)

train_df['no_store'] = train_df['store_count'].apply(lambda x: int(x == 0))
train_df


puchases_train['day_of_week'] = puchases_train['transaction_datetime'].dt.dayofweek

dop = puchases_train.groupby('client_id')['day_of_week'].agg(lambda x: x.value_counts().idxmax()).to_frame()

train_df = pd.merge(train_df, dop, on='client_id', how='inner')
puchases_train['is_friday'] = puchases_train['day_of_week'].apply(lambda x: int(x == 4))

friday = puchases_train.drop_duplicates('transaction_id').groupby('client_id')['is_friday'].mean()

train_df['friday'] = train_df['client_id'].map(friday)
train_df['friday_33'] = (train_df['friday'] >= 0.33).astype('int')
dop = puchases_train.groupby('client_id')['purchase_sum'].agg(lambda x: x.value_counts().idxmax()).to_frame()

train_df = pd.merge(train_df, dop, on='client_id', how='inner')
train_df['more_500'] = (train_df['purchase_sum'] >= 500).astype('int')
dop = clients.groupby('client_id')['age'].agg(lambda x: x.value_counts().idxmax()).to_frame()

train_df = pd.merge(train_df, dop, on='client_id', how='inner')
dop = clients.groupby('client_id')['gender'].agg(lambda x: x.value_counts().idxmax()).to_frame()

train_df = pd.merge(train_df, dop, on='client_id', how='inner')
train_df['pensioner_F']=((train_df['age'] >=60) & (train_df['gender']=='F')).astype('int')
most_popular_level3 = product_df['level_3'].value_counts().idxmax()

puchases_train['popular_level3'] = product_df['level_3'].apply(lambda x: int(x == most_popular_level3))

most_pop_level3 = puchases_train.groupby('client_id')['popular_level3'].sum()

train_df['level_3'] = train_df['client_id'].map(most_pop_level3)
train_df['level_3'].sum()
train_df
# создаем трейн и валидацию

cols = ['no_store','friday_33','more_500','pensioner_F','level_3']

x_train, x_valid, y_train, y_valid = train_test_split(train_df[cols], train_df['new_target'], test_size=.2)
# обучение модели

params = {

#     'boosting_type': 'gbdt',

    'n_estimators': 1000, # максимальное количество деревьев

    'max_depth': 2, # глубина одного дерева

    'learning_rate' : 0.1, # скорость обучения

    'num_leaves': 3, # количество листьев в дереве

    'min_data_in_leaf': 50, # минимальное количество наблюдений в листе

    'lambda_l1': 0, # параметр регуляризации

    'lambda_l2': 0, # параметр регуляризации

    

    'early_stopping_rounds': 20, # количество итераций без улучшения целевой метрики

}



lgbm = LGBMClassifier(**params)



lgbm = lgbm.fit(x_train, y_train, verbose=50, eval_set=[(x_valid, y_valid)], eval_metric='AUC')

predicts = lgbm.predict_proba(x_valid)[:, 1]



#gini

2*roc_auc_score(y_valid, predicts) - 1