import pandas as pd

import numpy as np



from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()
train = pd.read_csv('../../kaggle/input/x5-uplift-valid/data/train.csv', sep=',', encoding='cp1251')

train_purch = pd.read_csv('../../kaggle/input/x5-uplift-valid/train_purch/train_purch.csv', sep=',', encoding='cp1251')

clients = pd.read_csv('../../kaggle/input/x5-uplift-valid/data/clients2.csv', sep=',', encoding='cp1251')

products = pd.read_csv('../../kaggle/input/x5-uplift-valid/data/products.csv', sep=',', encoding='cp1251')
train_prep = train.copy()
train_prep['new_target'] = 0

train_prep.loc[(train_prep['treatment_flg'] == 1) & (train_prep['target'] == 1), 'new_target'] = 1

train_prep.loc[(train_prep['treatment_flg'] == 0) & (train_prep['target'] == 0), 'new_target'] = 1
train_prep.head()
train_purch['transaction_datetime'] = pd.to_datetime(train_purch['transaction_datetime'])

train_purch_merged = pd.merge(train_purch, products, on='product_id', how='inner')
alcohol_sum = train_purch_merged.groupby('client_id')['is_alcohol'].sum()

train_prep['alcohol_count'] = train_prep['client_id'].map(alcohol_sum)

train_prep['no_alcohol'] = train_prep['alcohol_count'].apply(lambda x: int(x == 0))
train_prep.head()
most_popular_brand = train_purch_merged['brand_id'].value_counts().idxmax()

train_purch_merged['most_pop_brand'] = train_purch_merged['brand_id'].apply(lambda x: int(x == most_popular_brand))
most_pop_brand_counter = train_purch_merged.groupby('client_id')['most_pop_brand'].sum()

train_prep['brand_count'] = train_prep['client_id'].map(most_pop_brand_counter)

train_prep['no_brand'] = train_prep['brand_count'].apply(lambda x: int(x == 0))
most_popular_segment = train_purch_merged['segment_id'].value_counts().idxmax()

train_purch_merged['most_pop_segm'] = train_purch_merged['segment_id'].apply(lambda x: int(x == most_popular_segment))
most_pop_segm_counter = train_purch_merged.groupby('client_id')['most_pop_segm'].sum()

train_prep['segm_count'] = train_prep['client_id'].map(most_pop_segm_counter)

train_prep['no_segm'] = train_prep['segm_count'].apply(lambda x: int(x == 0))
unique_stores = train_purch_merged.drop_duplicates('transaction_id').groupby('client_id')['store_id'].nunique()

train_prep['unique_stores'] = train_prep['client_id'].map(unique_stores)

train_prep['unique_stores_2'] = train_prep['unique_stores'].apply(lambda x: int(x < 2))
transactions = train_purch_merged.drop_duplicates('transaction_id').groupby('client_id')['transaction_id'].nunique()

total_products = train_purch_merged.groupby('client_id')['product_quantity'].sum()

train_prep['transactions'] = train_prep['client_id'].map(transactions)

train_prep['total_products'] = train_prep['client_id'].map(total_products)

train_prep['avg_product_quantity']  = train_prep['total_products'] / train_prep['transactions']

train_prep['avg_product_quantity_10']  = train_prep['avg_product_quantity'].apply(lambda x: int(x < 10))
from sklearn.model_selection import train_test_split
cols = ['no_alcohol','no_brand','no_segm', 'unique_stores_2', 'avg_product_quantity_10']

x_train, x_valid, y_train, y_valid = train_test_split(train_prep[cols], train_prep['new_target'], test_size=.2)
import xgboost as xgb

from xgboost import XGBClassifier

from xgboost import plot_importance

from sklearn.metrics import roc_auc_score
params = {

    'n_estimators': 1000, # максимальное количество деревьев

    'max_depth': 3, # глубина одного дерева

    'learning_rate' : 0.1, # скорость обучения

    'num_leaves': 3, # количество листьев в дереве

    'min_data_in_leaf': 50, # минимальное количество наблюдений в листе

    'lambda_l1': 1, # параметр регуляризации

    'lambda_l2': 0, # параметр регуляризации

    

    'early_stopping_rounds': 10, # количество итераций без улучшения целевой метрики

}
xgb = XGBClassifier(**params)



xgb = xgb.fit(x_train, y_train, verbose=50, eval_set=[(x_valid, y_valid)], eval_metric='auc')

predicts = xgb.predict_proba(x_valid)[:, 1]



#gini

2*roc_auc_score(y_valid, predicts) - 1
plot_importance(xgb)

plt.show()
from scipy.stats import ttest_ind, ttest_rel
def get_samples(x_train, feature, n_iterations):

    group1 = y_train.loc[x_train[x_train[feature] == 1].index]

    group2 = y_train.loc[x_train[x_train[feature] == 0].index]

    

    group1_mean = []

    for x in range(n_iterations):

        samples = np.random.choice(group1, size=group1.shape[0], replace=True)

        group1_mean.append(samples.mean())

    

    group2_mean = []

    for x in range(n_iterations):

        samples = np.random.choice(group2, size=group2.shape[0], replace=True)

        group2_mean.append(samples.mean())

    return group1_mean, group2_mean
group1_mean, group2_mean = get_samples(x_train, 'no_alcohol', 5000)
plt.hist(group1_mean, bins=50, label='group1')

plt.hist(group2_mean, bins=50, label='group2')

plt.legend()

plt.show()
ttest_ind(group2_mean, group1_mean)
group1_mean, group2_mean = get_samples(x_train, 'no_brand', 5000)
plt.hist(group1_mean, bins=50, label='group1')

plt.hist(group2_mean, bins=50, label='group2')

plt.legend()

plt.show()
ttest_ind(group2_mean, group1_mean)
group1_mean, group2_mean = get_samples(x_train, 'no_segm', 5000)
plt.hist(group1_mean, bins=50, label='group1')

plt.hist(group2_mean, bins=50, label='group2')

plt.legend()

plt.show()
ttest_ind(group2_mean, group1_mean)
group1_mean, group2_mean = get_samples(x_train, 'unique_stores_2', 5000)
plt.hist(group1_mean, bins=50, label='group1')

plt.hist(group2_mean, bins=50, label='group2')

plt.legend()

plt.show()
ttest_ind(group2_mean, group1_mean)
group1_mean, group2_mean = get_samples(x_train, 'avg_product_quantity_10', 5000)
plt.hist(group1_mean, bins=50, label='group1')

plt.hist(group2_mean, bins=50, label='group2')

plt.legend()

plt.show()
ttest_ind(group2_mean, group1_mean)