import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
# sns.set(rc={'figure.figsize':(14.7,8.27)})
np.random.seed(sum(map(ord, "categorical")))

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


import os
print(os.listdir("../input"))

application_train = pd.read_csv("../input/application_train.csv")
application_test = pd.read_csv("../input/application_test.csv")
bureau = pd.read_csv("../input/bureau.csv")
bureau_balance = pd.read_csv("../input/bureau_balance.csv")
credit_card_balance = pd.read_csv("../input/credit_card_balance.csv")
installments_payments = pd.read_csv("../input/installments_payments.csv")
previous_application = pd.read_csv("../input/previous_application.csv")
POS_CASH_balance = pd.read_csv("../input/POS_CASH_balance.csv")
ax = sns.stripplot(x=application_train['AMT_INCOME_TOTAL'],palette="Set1", dodge=True, jitter=True)
plt.xticks(rotation=45)
# train = application_train.drop(application_train[(application_train['AMT_INCOME_TOTAL']>1.2)])
ax = sns.stripplot(x=application_train['AMT_CREDIT'],palette="Set1", dodge=True, jitter=True)
plt.xticks(rotation=45)
ax = sns.stripplot(x=application_train['AMT_GOODS_PRICE'],palette="Set1", dodge=True, jitter=True)
plt.xticks(rotation=45)
ax = sns.stripplot(x=application_train['AMT_ANNUITY'],palette="Set1", dodge=True, jitter=True)
plt.xticks(rotation=45)
application_train['FLOORSMIN_AVG'].unique()
print(application_train.columns, application_train.shape)
print(application_test.columns, application_test.shape)
print(bureau.columns, bureau.shape)
print(bureau_balance.columns, bureau_balance.shape)
print(credit_card_balance.columns, credit_card_balance.shape)
print(installments_payments.columns, installments_payments.shape)
print(previous_application.columns, previous_application.shape)
print(POS_CASH_balance.columns, POS_CASH_balance.shape)
sns.set(rc={'figure.figsize':(14.7,8.27)})
# family status
sns.barplot(x="NAME_FAMILY_STATUS", y="AMT_CREDIT", hue="TARGET", data=application_train, palette="icefire")
plt.xticks(rotation=45)
sns.countplot(x="NAME_FAMILY_STATUS", hue="TARGET", data=application_train, palette="icefire")
plt.xticks(rotation=45)
# amount of children
sns.barplot(x="CNT_CHILDREN", y="AMT_CREDIT", hue="TARGET", data=application_train, palette="icefire")
plt.xticks(rotation=45)
sns.countplot(x="CNT_CHILDREN", hue="TARGET", data=application_train, palette="icefire")
plt.xticks(rotation=45)
# how many family members
sns.barplot(x="CNT_FAM_MEMBERS", y="AMT_CREDIT", hue="TARGET", data=application_train, palette="icefire")
plt.xticks(rotation=45)
sns.countplot(x="CNT_FAM_MEMBERS", hue="TARGET", data=application_train, palette="icefire")
plt.xticks(rotation=45)
# education
sns.barplot(x="NAME_EDUCATION_TYPE", y="AMT_CREDIT", hue="TARGET", data=application_train, palette="icefire")
plt.xticks(rotation=45)
sns.countplot(x="NAME_EDUCATION_TYPE", hue="TARGET", data=application_train, palette="icefire")
plt.xticks(rotation=45)
# organization
sns.barplot(x="ORGANIZATION_TYPE", y="AMT_CREDIT", hue="TARGET", data=application_train, palette="icefire")
plt.xticks(rotation=45)
sns.countplot(x="ORGANIZATION_TYPE", hue="TARGET", data=application_train, palette="icefire")
plt.xticks(rotation=45)
plt.figure(figsize=(20,20))
sns.barplot(x="NAME_CONTRACT_TYPE", y="AMT_CREDIT", hue="TARGET", data=application_train, palette="icefire")
plt.xticks(rotation=45)
sns.countplot(x="NAME_CONTRACT_TYPE", hue="TARGET", data=application_train, palette="icefire")
plt.xticks(rotation=45)
# bureau.columns.values
bureau_cat = [f_ for f_ in bureau.columns if bureau[f_].dtype == 'object']
dummy_bureau = pd.get_dummies(bureau, columns=bureau_cat)
dummy_bureau.head()
bureau_balance_cat = [f_ for f_ in bureau_balance.columns if bureau_balance[f_].dtype == 'object']
dummy_bureau_balance = pd.get_dummies(bureau_balance, columns=bureau_balance_cat)
dummy_bureau_balance.head()
avg_bureau_balance = dummy_bureau_balance.groupby('SK_ID_BUREAU').mean()
avg_bureau_balance.head()
bureau_all = dummy_bureau.merge(right=avg_bureau_balance.reset_index(), how='left', on='SK_ID_BUREAU', suffixes=('', '_balance_'))
bureau_all.head()
bureau_per_curr = bureau_all[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()
bureau_per_curr.head(10)
bureau_all['SK_ID_BUREAU'] = bureau_all['SK_ID_CURR'].map(bureau_per_curr['SK_ID_BUREAU'])
avg_bureau = bureau_all.groupby('SK_ID_CURR').mean()
avg_bureau.head(10)
del avg_bureau_balance, dummy_bureau, dummy_bureau_balance, bureau_all
credit_card_balance_cat = [f_ for f_ in credit_card_balance.columns if credit_card_balance[f_].dtype == 'object']
dummy_credit_card_balance = pd.get_dummies(credit_card_balance, columns=credit_card_balance_cat)
dummy_credit_card_balance.head()
credit_card_per_curr = dummy_credit_card_balance[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
dummy_credit_card_balance['SK_ID_PREV'] = dummy_credit_card_balance['SK_ID_CURR'].map(credit_card_per_curr['SK_ID_PREV'])
avg_credit_card = dummy_credit_card_balance.groupby('SK_ID_CURR').mean()
avg_credit_card.head()
del dummy_credit_card_balance
installments_payments_cat = [f_ for f_ in installments_payments.columns if installments_payments[f_].dtype == 'object']
dummy_installments_payments = pd.get_dummies(installments_payments, columns=installments_payments_cat)
dummy_installments_payments.head()
installments_per_curr = dummy_installments_payments[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
dummy_installments_payments['SK_ID_PREV'] = dummy_installments_payments['SK_ID_CURR'].map(installments_per_curr['SK_ID_PREV'])
avg_installments = dummy_installments_payments.groupby('SK_ID_CURR').mean()
avg_installments.head()
del dummy_installments_payments
pos_cash_balance_cat = [f_ for f_ in POS_CASH_balance.columns if POS_CASH_balance[f_].dtype == 'object']
dummy_POS_CASH_balance = pd.get_dummies(POS_CASH_balance, columns=pos_cash_balance_cat)
dummy_POS_CASH_balance.head()
pos_per_curr = dummy_POS_CASH_balance[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
dummy_POS_CASH_balance['SK_ID_PREV'] = dummy_POS_CASH_balance['SK_ID_CURR'].map(pos_per_curr['SK_ID_PREV'])
avg_pos = dummy_POS_CASH_balance.groupby('SK_ID_CURR').mean()
avg_pos.head()

del dummy_POS_CASH_balance
previous_application_cat = [f_ for f_ in previous_application.columns if previous_application[f_].dtype == 'object']
dummy_previous_application = pd.get_dummies(previous_application, columns=previous_application_cat)
dummy_previous_application.head()
previous_per_curr = dummy_previous_application[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
dummy_previous_application['SK_ID_PREV'] = dummy_previous_application['SK_ID_CURR'].map(previous_per_curr['SK_ID_PREV'])
dummy_previous_application.head(10)
avg_previous = dummy_previous_application.groupby('SK_ID_CURR').mean()
avg_previous.head()
del dummy_previous_application
del previous_application
y = application_train['TARGET']
del application_train['TARGET']
cat = [f for f in application_train.columns if application_train[f].dtype == 'object']
for f_ in cat:
    application_train[f_], indexer = pd.factorize(application_train[f_])
    application_test[f_] = indexer.get_indexer(application_test[f_])
    
for f_ in cat:
    print('{}: {}'.format(f_, application_train[f_].unique())) 
    
application_train.head()    
import missingno as msno
# missing features correlation matrix
msno.heatmap(application_train)
# msno.matrix(application_train)
application_train.isnull().sum().sort_values(ascending=False)
(application_train.isnull().mean() > 0.6).sum()
application_train = application_train.merge(right=avg_bureau.reset_index(), how='left', on='SK_ID_CURR')
application_test = application_test.merge(right=avg_bureau.reset_index(), how='left', on='SK_ID_CURR')

application_train = application_train.merge(right=avg_previous.reset_index(), how='left', on='SK_ID_CURR')
application_test = application_test.merge(right=avg_previous.reset_index(), how='left', on='SK_ID_CURR')

application_train = application_train.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')
application_test = application_test.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')
application_train = application_train.merge(right=avg_installments.reset_index(), how='left', on='SK_ID_CURR')
application_test = application_test.merge(right=avg_installments.reset_index(), how='left', on='SK_ID_CURR')
application_train = application_train.merge(right=avg_credit_card.reset_index(), how='left', on='SK_ID_CURR')
application_test = application_test.merge(right=avg_credit_card.reset_index(), how='left', on='SK_ID_CURR')
ax = sns.countplot(y,label="Count")
R, N = y.value_counts()
print('Rely: ',R)
print('Not rely : ',N)
# model and parameters
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X = application_train
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
# from sklearn.impute import SimpleImputer
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.pipeline import Pipeline
# # imputer = Imputer()
# estimator = Pipeline([("impute", SimpleImputer(strategy="mean")),
#                       ("scale", StandardScaler()),
#                       ("forest", RandomForestRegressor(random_state=0,
#                                                        n_estimators=100))])

# estimator.fit(x_train, y_train)
# x_train = imputer.fit_transform(x_train)
# x_test = imputer.fit_transform(x_test)
import lightgbm as lgb
train_data=lgb.Dataset(x_train,label=y_train)
test_data=lgb.Dataset(x_test,label=y_test)
#define paramters
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},
    'metric_freq': 1,
    'is_training_metric': True,
    'max_bin': 255,
    'learning_rate': 0.1,
    'num_leaves': 63,
    'tree_learner': 'serial',
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 50,
    'min_sum_hessian_in_leaf': 5,
    'is_enable_sparse': True,
    'use_two_round_loading': False,
    'is_save_binary_file': False,
    'num_machines': 1,
    'verbose': 0,
    'subsample_for_bin': 200000,
    'min_child_samples': 20,
    'min_child_weight': 0.001,
    'min_split_gain': 0.0,
    'colsample_bytree': 1.0,
    'reg_alpha': 0.0,
    'reg_lambda': 0.0
}
clf = lgb.train(params, train_data, 2000, valid_sets=test_data, early_stopping_rounds= 40, verbose_eval= 10)
y_prediction=clf.predict(application_train)

#Accuracy
from sklearn.metrics import roc_auc_score
score = roc_auc_score(y, y_prediction)
print("Overall AUC: {:.3f}" .format(score))
# submit
submit = clf.predict(application_test)
application_test['TARGET'] = submit
application_test[['SK_ID_CURR', 'TARGET']].to_csv('submission.csv', index=False, float_format='%.8f')