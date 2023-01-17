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
train_purch_merged.columns
train_purch_merged['day_of_week'] = train_purch_merged['transaction_datetime'].dt.dayofweek
temp_df = train_purch_merged.groupby('client_id')['day_of_week'].agg(lambda x: x.value_counts().idxmax()).to_frame()
train_prep = pd.merge(train_prep, temp_df, on='client_id', how='inner')
train_purch_merged['is_monday'] = train_purch_merged['day_of_week'].apply(lambda x: int(x == 0))
purchases_monday = train_purch_merged.drop_duplicates('transaction_id').groupby('client_id')['is_monday'].mean()
train_prep['monday_share'] = train_prep['client_id'].map(purchases_monday)
train_prep['monday_share_25'] = train_prep['monday_share'].apply(lambda x: int(x >= 0.25))
train_prep['monday_share_25'].value_counts()
trn_sum = train_purch_merged.groupby('client_id')['trn_sum_from_iss'].sum()
train_prep['trn_sum'] = train_prep['client_id'].map(trn_sum)
median = train_prep['trn_sum'].median()
train_prep['trn_sum_median'] = train_prep['trn_sum'].apply(lambda x: int(x > median))

train_prep['trn_sum_median'].value_counts()
most_popular_level1 = train_purch_merged['level_1'].value_counts().idxmax()
train_purch_merged['most_pop_level1'] = train_purch_merged['level_1'].apply(lambda x: int(x == most_popular_level1))
most_pop_level1 = train_purch_merged.groupby('client_id')['most_pop_level1'].sum()
train_prep['level1_count'] = train_prep['client_id'].map(most_pop_level1)
median = train_prep['level1_count'].median()
train_prep['level1_count_m'] = train_prep['level1_count'].apply(lambda x: int(x > median))
train_prep['level1_count_m'].value_counts()
most_popular_level2 = train_purch_merged['level_2'].value_counts().idxmax()
train_purch_merged['most_pop_level2'] = train_purch_merged['level_2'].apply(lambda x: int(x == most_popular_level2))
most_pop_level2 = train_purch_merged.groupby('client_id')['most_pop_level2'].sum()
train_prep['level2_count'] = train_prep['client_id'].map(most_pop_level2)
median = train_prep['level2_count'].mean()/2
train_prep['level2_count_m'] = train_prep['level2_count'].apply(lambda x: int(x > median))
train_prep['level2_count_m'].value_counts()
exp_points_received = train_purch_merged.groupby('client_id')['regular_points_received'].sum()
train_prep['reg_points_received'] = train_prep['client_id'].map(exp_points_received)
train_prep['reg_points_received_50'] = train_prep['reg_points_received'].apply(lambda x: int(x > 50))
train_prep['reg_points_received_50'].value_counts()
from sklearn.model_selection import train_test_split
cols = ['monday_share_25','trn_sum_median','level1_count_m', 'level2_count_m', 'reg_points_received_50']
x_train, x_valid, y_train, y_valid = train_test_split(train_prep[cols], train_prep['new_target'], test_size=.2)
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import roc_auc_score
params = {
    'n_estimators': 1000, # максимальное количество деревьев
    'max_depth': 2, # глубина одного дерева
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
group1_mean, group2_mean = get_samples(x_train, 'monday_share_25', 5000)
plt.hist(group1_mean, bins=50, label='group1')
plt.hist(group2_mean, bins=50, label='group2')
plt.legend()
plt.show()
ttest_ind(group2_mean, group1_mean)
group1_mean, group2_mean = get_samples(x_train, 'trn_sum_median', 5000)
plt.hist(group1_mean, bins=50, label='group1')
plt.hist(group2_mean, bins=50, label='group2')
plt.legend()
plt.show()
ttest_ind(group2_mean, group1_mean)
group1_mean, group2_mean = get_samples(x_train, 'level1_count_m', 5000)
plt.hist(group1_mean, bins=50, label='group1')
plt.hist(group2_mean, bins=50, label='group2')
plt.legend()
plt.show()
ttest_ind(group2_mean, group1_mean)
group1_mean, group2_mean = get_samples(x_train, 'level2_count_m', 5000)
plt.hist(group1_mean, bins=50, label='group1')
plt.hist(group2_mean, bins=50, label='group2')
plt.legend()
plt.show()
ttest_ind(group2_mean, group1_mean)
group1_mean, group2_mean = get_samples(x_train, 'reg_points_received_50', 5000)
plt.hist(group1_mean, bins=50, label='group1')
plt.hist(group2_mean, bins=50, label='group2')
plt.legend()
plt.show()
ttest_ind(group2_mean, group1_mean)
