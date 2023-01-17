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
# средний чек

avr_chech_sum_train = puchases_train.drop_duplicates('transaction_id').groupby('client_id')['purchase_sum'].mean()

avr_chech_sum_test = puchases_test.drop_duplicates('transaction_id').groupby('client_id')['purchase_sum'].mean()



train_df['avr_chech_sum'] = train_df['client_id'].map(avr_chech_sum_train)

test_df['avr_chech_sum'] = test_df['client_id'].map(avr_chech_sum_test)



# возраст

train_df['age'] = train_df['client_id'].map(clients.set_index('client_id')['age'])

test_df['age'] = test_df['client_id'].map(clients.set_index('client_id')['age'])



train_df['is_age_bigger_than_53'] = (train_df['age'] > 53).astype('int')

test_df['is_age_bigger_than_53'] = (test_df['age'] > 53).astype('int')



# алкоголь в покупках

mapper = product_df.set_index('product_id')['is_alcohol']

puchases_train['is_alcohol'] = puchases_train['product_id'].map(mapper)

puchases_test['is_alcohol'] = puchases_test['product_id'].map(mapper)



alco_mapper_train = puchases_train.groupby('client_id')['is_alcohol'].sum()

train_df['alco_produts_count'] = train_df['client_id'].map(alco_mapper_train)



alco_mapper_test = puchases_test.groupby('client_id')['is_alcohol'].sum()

test_df['alco_produts_count'] = test_df['client_id'].map(alco_mapper_test)



train_df['is_alco_in_check'] = (train_df['alco_produts_count'] > 0).astype('int')

test_df['is_alco_in_check'] = (test_df['alco_produts_count'] > 0).astype('int')
# создаем трейн и валидацию

cols = ['age', 'avr_chech_sum', 'is_alco_in_check', 'is_age_bigger_than_53']

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
# предикт и запись в файл

test_df['pred'] = lgbm.predict_proba(test_df[cols])[:, 1]

test_df.set_index('client_id')[['pred']].to_csv('baseline.csv')
median_sum = x_train['avr_chech_sum'].median()
x_train['is_avr_chech_sum_bigger_than_median_sum'] = (x_train['avr_chech_sum'] > median_sum).astype('int')
feature = 'is_alco_in_check'

x_train[feature].value_counts(1)
group1 = y_train.loc[x_train[x_train[feature] == 1].index]

group2 = y_train.loc[x_train[x_train[feature] == 0].index]
%%time

n_iterations = 5000

group1_mean = []

for x in range(n_iterations):

    samples = np.random.choice(group1, size=group1.shape[0], replace=True)

    group1_mean.append(samples.mean())
%%time

group2_mean = []

for x in range(n_iterations):

    samples = np.random.choice(group2, size=group2.shape[0], replace=True)

    group2_mean.append(samples.mean())
plt.hist(group1_mean, bins=50, label='group1')

plt.hist(group2_mean, bins=50, label='group2')

plt.legend()

plt.show()
ttest_ind(group2_mean, group1_mean)
params = {

#     'boosting_type': 'gbdt',

    'n_estimators': 1000, # максимальное количество деревьев

    'max_depth': 2, # глубина одного дерева

    'learning_rate' : 0.1, # скорость обучения

    'num_leaves': 3, # количество листьев в дереве

    'min_data_in_leaf': 50, # минимальное количество наблюдений в листе

    'lambda_l1': 0, # параметр регуляризации

    'lambda_l2': 0, # параметр регуляризации

    'verbose': 0,

    

    'early_stopping_rounds': 20, # количество итераций без улучшения целевой метрики

}
# создаем трейн и валидацию

cols = ['age', 'avr_chech_sum', 'is_alco_in_check']

x_train, x_valid, y_train, y_valid = train_test_split(train_df[cols], train_df['new_target'], test_size=.2)
skf = StratifiedKFold(n_splits=50, random_state=17, shuffle=True)
def calc_scores(features_list):



    scores = []



    for trn_ind, val_ind in skf.split(train_df[features_list], train_df['new_target']):



        train_data = train_df.iloc[trn_ind][features_list]

        valid_data = train_df.iloc[val_ind][features_list]



        train_target = train_df.iloc[trn_ind]['new_target']

        valid_target = train_df.iloc[val_ind]['new_target']



        lgbm = LGBMClassifier(**params)



        lgbm = lgbm.fit(train_data, train_target, verbose=0, eval_set=[(valid_data, valid_target)], eval_metric='AUC')

        predicts = lgbm.predict_proba(valid_data)[:, 1]



        score = roc_auc_score(valid_target, predicts)

        scores.append(score)

        

    return scores
scores1 = calc_scores(['age', 'avr_chech_sum', 'is_alco_in_check'])
scores2 = calc_scores(['age', 'avr_chech_sum'])
np.mean(scores1), np.std(scores1)
np.mean(scores2), np.std(scores2)
ttest_rel(scores1, scores2)