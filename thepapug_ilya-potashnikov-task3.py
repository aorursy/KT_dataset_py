import numpy as np

import pandas as pd

from datetime import datetime

import os

from datetime import timedelta

import matplotlib.pyplot as plt
df = pd.read_csv('../../kaggle/input/x5-uplift-valid/data/train.csv', sep=',', encoding='cp1251')

purchases = pd.read_csv('../../kaggle/input/x5-uplift-valid/train_purch/train_purch.csv', sep=',', encoding='cp1251')

clients = pd.read_csv('../../kaggle/input/x5-uplift-valid/data/clients2.csv', sep=',', encoding='cp1251')

products = pd.read_csv('../../kaggle/input/x5-uplift-valid/data/products.csv', sep=',', encoding='cp1251')
def calc_uplift(df):

    sms_sent = df[df['treatment_flg'] == 1]

    response_sms_sent = df[(df['treatment_flg'] == 1) & (df['target'] == 1)]

    pr_sms = len(response_sms_sent)/len(sms_sent)

    no_sms = df[df['treatment_flg'] == 0]

    response_no_sms = df[(df['treatment_flg'] == 0) & (df['target'] == 1)]

    pr_no_sms = len(response_no_sms)/len(no_sms)

    uplift_score = pr_sms - pr_no_sms

    return uplift_score
purchases_merged = pd.merge(purchases, products, on='product_id', how='inner')
train_feature = df.copy()
netto = purchases_merged.groupby('client_id')['netto'].sum()

train_feature['netto'] = train_feature['client_id'].map(netto)

thrsh = train_feature['netto'].median()

train_feature['netto_57'] = train_feature['netto'].apply(lambda x: int(x > thrsh))
exp_points_spent = purchases_merged.groupby('client_id')['express_points_spent'].sum()

train_feature['exp_points_spent'] = train_feature['client_id'].map(exp_points_spent)

train_feature['exp_points_spent_not_null'] = train_feature['exp_points_spent'].apply(lambda x: int(x != 0))
purchases_merged['transaction_datetime'] = pd.to_datetime(purchases_merged['transaction_datetime'])

purchases_merged['hour'] = purchases_merged['transaction_datetime'].apply(lambda x: x.hour)
hour = purchases_merged.groupby('client_id')['hour'].mean()

train_feature['hour'] = train_feature['client_id'].map(hour)

train_feature['day'] = train_feature['hour'].apply(lambda x: int(x > 12 and x <18))
train_feature['morning'] = train_feature['hour'].apply(lambda x: int(x <= 12))
most_popular_vendor_id = products['vendor_id'].value_counts().idxmax()

purchases_merged['most_pop_vend'] = purchases_merged['vendor_id'].apply(lambda x: int(x == most_popular_vendor_id))
most_pop_vend = purchases_merged.groupby('client_id')['most_pop_vend'].sum()

train_feature['vendor_count'] = train_feature['client_id'].map(most_pop_vend)

train_feature['no_vendor'] = train_feature['vendor_count'].apply(lambda x: int(x == 0))
train_feature['new_target'] = 0

train_feature.loc[(train_feature['treatment_flg'] == 1) & (train_feature['target'] == 1), 'new_target'] = 1

train_feature.loc[(train_feature['treatment_flg'] == 0) & (train_feature['target'] == 0), 'new_target'] = 1
import xgboost as xgb

from xgboost import XGBClassifier

from xgboost import plot_importance

from sklearn.metrics import roc_auc_score

from matplotlib import pyplot as plt
params = {

    'n_estimators': 1000, # максимальное количество деревьев

    'max_depth': 3, # глубина одного дерева

    'learning_rate' : 0.1, # скорость обучения

    'num_leaves': 3, # количество листьев в дереве

    'min_data_in_leaf': 50, # минимальное количество наблюдений в листе

    'lambda_l1': 0, # параметр регуляризации

    'lambda_l2': 0, # параметр регуляризации

    

    'early_stopping_rounds': 20, # количество итераций без улучшения целевой метрики

}
xgb = XGBClassifier(**params)
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_feature['new_target'], test_size=.2)



cols = ['netto_57', 'exp_points_spent_not_null', 'day', 'morning', 'no_vendor']

x_train_cut = x_train[cols]

x_valid_cut = x_valid[cols]
xgb = xgb.fit(x_train_cut, y_train, verbose=50, eval_set=[(x_valid_cut, y_valid)], eval_metric='auc')

predicts = xgb.predict_proba(x_valid_cut)[:, 1]
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
x_train['netto_57'].value_counts()
group1_mean, group2_mean = get_samples(x_train, 'netto_57', 5000)
plt.hist(group1_mean, bins=50, label='group1')

plt.hist(group2_mean, bins=50, label='group2')

plt.legend()

plt.show()
ttest_ind(group2_mean, group1_mean)
group1_mean, group2_mean = get_samples(x_train, 'exp_points_spent_not_null', 5000)
plt.hist(group1_mean, bins=50, label='group1')

plt.hist(group2_mean, bins=50, label='group2')

plt.legend()

plt.show()
ttest_ind(group2_mean, group1_mean)
group1_mean, group2_mean = get_samples(x_train, 'day', 5000)
plt.hist(group1_mean, bins=50, label='group1')

plt.hist(group2_mean, bins=50, label='group2')

plt.legend()

plt.show()
ttest_ind(group2_mean, group1_mean)
group1_mean, group2_mean = get_samples(x_train, 'morning', 5000)
plt.hist(group1_mean, bins=50, label='group1')

plt.hist(group2_mean, bins=50, label='group2')

plt.legend()

plt.show()
ttest_ind(group2_mean, group1_mean)
group1_mean, group2_mean = get_samples(x_train, 'no_vendor', 5000)
plt.hist(group1_mean, bins=50, label='group1')

plt.hist(group2_mean, bins=50, label='group2')

plt.legend()

plt.show()
ttest_ind(group2_mean, group1_mean)