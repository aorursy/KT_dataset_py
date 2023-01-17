import numpy as np
import pandas as pd
import seaborn as sns

import datetime
import lightgbm as lgb
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import os
import xgboost as xgb
import lightgbm as lgb

import os
import matplotlib.pyplot as plt
%matplotlib inline
pd.set_option('max_columns', 150)
folder = '../input/'
application_train = pd.read_csv(os.path.join(folder, 'application_train.csv'))
application_test = pd.read_csv(os.path.join(folder, 'application_test.csv'))
bureau = pd.read_csv(os.path.join(folder, 'bureau.csv'))
bureau_balance = pd.read_csv(os.path.join(folder, 'bureau_balance.csv'))
POS_CASH_balance = pd.read_csv(os.path.join(folder, 'POS_CASH_balance.csv'))
credit_card_balance = pd.read_csv(os.path.join(folder, 'credit_card_balance.csv'))
previous_application = pd.read_csv(os.path.join(folder, 'previous_application.csv'))
installments_payments = pd.read_csv(os.path.join(folder, 'installments_payments.csv'))
sample_submission = pd.read_csv(os.path.join(folder, 'sample_submission.csv'))
application_train.head()
#application_train.shape
application_train.TARGET.value_counts(normalize=True)
pd.crosstab(application_train.TARGET, application_train.NAME_CONTRACT_TYPE, dropna=False, normalize='all')
pd.crosstab(application_train.TARGET, application_train.CODE_GENDER, dropna=False)
application_train.groupby(["FLAG_OWN_CAR", "OWN_CAR_AGE"]).mean().head()
application_train.OWN_CAR_AGE.mean()
print('There are {0} people with realty. {1}% of them repay loans.'.format(application_train[application_train.FLAG_OWN_REALTY == 'Y'].shape[0], np.round(application_train[application_train.FLAG_OWN_REALTY == 'Y'].TARGET.value_counts(normalize=True).values[1], 3) * 100))
print('There are {0} people with cars. {1}% of them repay loans.'.format(application_train[application_train.FLAG_OWN_CAR == 'Y'].shape[0], np.round(application_train[application_train.FLAG_OWN_CAR == 'Y'].TARGET.value_counts(normalize=True).values[1], 4) * 100))
print('Average age of the car is {:.2f} years.'.format(application_train.OWN_CAR_AGE.mean()))
pd.crosstab(application_train.CNT_CHILDREN, application_train.NAME_FAMILY_STATUS, dropna=False)
pd.crosstab(application_train.CNT_CHILDREN, application_train.CNT_FAM_MEMBERS, dropna=False)
application_train['NAME_TYPE_SUITE'].value_counts(dropna=False)
pd.crosstab(application_train.NAME_TYPE_SUITE, application_train.NAME_FAMILY_STATUS, dropna=False)
application_train.groupby(['NAME_INCOME_TYPE']).agg({'AMT_INCOME_TOTAL': ['mean', 'median', 'count']}).reset_index().sort_index()
application_train[application_train['NAME_INCOME_TYPE'] == 'Maternity leave']['CODE_GENDER'].value_counts()

s = pd.crosstab(application_train.NAME_INCOME_TYPE, application_train.OCCUPATION_TYPE, dropna=False).style.background_gradient(cmap='viridis', low=.5, high=0).highlight_null('red')
s
print('{0} zero values.'.format(application_train[application_train['AMT_GOODS_PRICE'].isnull()].shape[0]))
non_zero_good_price = application_train[application_train['AMT_GOODS_PRICE'].isnull() == False]
credit_to_good_price = non_zero_good_price['AMT_CREDIT'] / non_zero_good_price['AMT_GOODS_PRICE']
#credit_to_good_price
plt.boxplot(credit_to_good_price);
plt.title('Credit amount to goods price.');
sns.countplot(application_train['NAME_HOUSING_TYPE']);
plt.xticks(rotation=45);
plt.title('Counts of housing type')
application_train['contact_info'] = application_train['FLAG_MOBIL'] + application_train['FLAG_EMP_PHONE'] + application_train['FLAG_WORK_PHONE'] + application_train['FLAG_CONT_MOBILE'] + application_train['FLAG_PHONE'] + application_train['FLAG_EMAIL']
sns.countplot(application_train['contact_info']);
plt.title('Count of ways to contact client');
application_train.loc[application_train['OBS_30_CNT_SOCIAL_CIRCLE'] > 1, 'OBS_30_CNT_SOCIAL_CIRCLE'] = '1+'
application_train.loc[application_train['DEF_30_CNT_SOCIAL_CIRCLE'] > 1, 'DEF_30_CNT_SOCIAL_CIRCLE'] = '1+'
application_train.loc[application_train['OBS_60_CNT_SOCIAL_CIRCLE'] > 1, 'OBS_60_CNT_SOCIAL_CIRCLE'] = '1+'
application_train.loc[application_train['DEF_60_CNT_SOCIAL_CIRCLE'] > 1, 'DEF_60_CNT_SOCIAL_CIRCLE'] = '1+'
#application_train.head()

fig, ax = plt.subplots(figsize = (30, 8))
plt.subplot(1, 4, 1)
sns.countplot(application_train['OBS_30_CNT_SOCIAL_CIRCLE']);
plt.subplot(1, 4, 2)
sns.countplot(application_train['DEF_30_CNT_SOCIAL_CIRCLE']);
plt.subplot(1, 4, 3)
sns.countplot(application_train['OBS_60_CNT_SOCIAL_CIRCLE']);
plt.subplot(1, 4, 4)
sns.countplot(application_train['DEF_60_CNT_SOCIAL_CIRCLE']);
sns.boxplot(application_train['AMT_INCOME_TOTAL']);
plt.title('AMT_INCOME_TOTAL boxplot');
sns.boxplot(application_train[application_train['AMT_INCOME_TOTAL'] < np.percentile(application_train['AMT_INCOME_TOTAL'], 90)]['AMT_INCOME_TOTAL']);
plt.title('AMT_INCOME_TOTAL boxplot on data within 90 percentile');
application_train.groupby('TARGET').agg({'AMT_INCOME_TOTAL': ['mean', 'median', 'count']}).reset_index()
plt.hist(application_train['AMT_INCOME_TOTAL']);
plt.title('AMT_INCOME_TOTAL histogram');
plt.hist(application_train[application_train['AMT_INCOME_TOTAL'] < np.percentile(application_train['AMT_INCOME_TOTAL'], 90)]['AMT_INCOME_TOTAL']);
plt.title('AMT_INCOME_TOTAL histogram on data within 90 percentile');
plt.hist(np.log1p(application_train['AMT_INCOME_TOTAL']));
plt.title('AMT_INCOME_TOTAL histogram on data with log1p transformation');
sns.boxplot(application_train['AMT_CREDIT'], orient='v');
plt.title('AMT_CREDIT boxplot');
sns.boxplot(application_train[application_train['AMT_CREDIT'] < np.percentile(application_train['AMT_CREDIT'], 95)]['AMT_CREDIT'], orient='v');
plt.title('AMT_CREDIT boxplot on data within 90 percentile');
application_train.groupby('TARGET').agg({'AMT_CREDIT': ['mean', 'median', 'count']})
plt.hist(application_train['AMT_CREDIT']);
plt.title('AMT_CREDIT histogram');
plt.hist(application_train[application_train['AMT_CREDIT'] < np.percentile(application_train['AMT_CREDIT'], 90)]['AMT_CREDIT']);
plt.title('AMT_INCOME_TOTAL histogram on data within 90 percentile');
plt.hist(np.log1p(application_train['AMT_CREDIT']));
plt.title('AMT_CREDIT histogram on data with log1p transformation');
application_train['age'] = application_train['DAYS_BIRTH'] / -365
plt.hist(application_train['age']);
plt.title('Histogram of age in years.');
application_train.loc[application_train['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = 0
application_train['years_employed'] = application_train['DAYS_EMPLOYED'] / -365
plt.hist(application_train['years_employed']);
plt.title('Length of working at current workplace in years.');
application_train.groupby(['NAME_INCOME_TYPE']).agg({'years_employed': ['mean', 'median', 'count', 'max'], 'age': ['median']})
application_train.groupby(['NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE']).agg({'AMT_INCOME_TOTAL': ['mean', 'median', 'count', 'max']})
application_train['AMT_INCOME_TOTAL'] = np.log1p(application_train['AMT_INCOME_TOTAL'])
application_train['AMT_CREDIT'] = np.log1p(application_train['AMT_CREDIT'])
application_train['OWN_CAR_AGE'] = application_train['OWN_CAR_AGE'].fillna(0)
application_train['app AMT_CREDIT / AMT_ANNUITY'] = application_train['AMT_CREDIT'] / application_train['AMT_ANNUITY']
application_train['app EXT_SOURCE mean'] = application_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis = 1)
application_train['app EXT_SOURCE_1 / DAYS_BIRTH'] = application_train['EXT_SOURCE_1'] / application_train['DAYS_BIRTH']
application_train['app AMT_INCOME_TOTAL / 12 - AMT_ANNUITY'] = application_train['AMT_INCOME_TOTAL'] / 12. - application_train['AMT_ANNUITY']
application_train['app AMT_INCOME_TOTAL / AMT_ANNUITY'] = application_train['AMT_INCOME_TOTAL'] / application_train['AMT_ANNUITY']
application_train['app AMT_INCOME_TOTAL - AMT_GOODS_PRICE'] = application_train['AMT_INCOME_TOTAL'] - application_train['AMT_GOODS_PRICE']
application_test.loc[application_test['OBS_30_CNT_SOCIAL_CIRCLE'] > 1, 'OBS_30_CNT_SOCIAL_CIRCLE'] = '1+'
application_test.loc[application_test['DEF_30_CNT_SOCIAL_CIRCLE'] > 1, 'DEF_30_CNT_SOCIAL_CIRCLE'] = '1+'
application_test.loc[application_test['OBS_60_CNT_SOCIAL_CIRCLE'] > 1, 'OBS_60_CNT_SOCIAL_CIRCLE'] = '1+'
application_test.loc[application_test['DEF_60_CNT_SOCIAL_CIRCLE'] > 1, 'DEF_60_CNT_SOCIAL_CIRCLE'] = '1+'
np.log1p(application_test['AMT_INCOME_TOTAL'])
np.log1p(application_test['AMT_CREDIT'])
application_test['age'] = application_test['DAYS_BIRTH'] / -365
application_test.loc[application_test['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = 0
application_test['years_employed'] = application_test['DAYS_EMPLOYED'] / -365
application_test['AMT_INCOME_TOTAL'] = np.log1p(application_test['AMT_INCOME_TOTAL'])
application_test['AMT_CREDIT'] = np.log1p(application_test['AMT_CREDIT'])
application_test['OWN_CAR_AGE'] = application_test['OWN_CAR_AGE'].fillna(0)
application_test['app AMT_CREDIT / AMT_ANNUITY'] = application_test['AMT_CREDIT'] / application_test['AMT_ANNUITY']
application_test['app EXT_SOURCE mean'] = application_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis = 1)
application_test['app EXT_SOURCE_1 / DAYS_BIRTH'] = application_test['EXT_SOURCE_1'] / application_test['DAYS_BIRTH']
application_test['app AMT_INCOME_TOTAL / 12 - AMT_ANNUITY'] = application_test['AMT_INCOME_TOTAL'] / 12. - application_test['AMT_ANNUITY']
application_test['app AMT_INCOME_TOTAL / AMT_ANNUITY'] = application_test['AMT_INCOME_TOTAL'] / application_test['AMT_ANNUITY']
application_test['app AMT_INCOME_TOTAL - AMT_GOODS_PRICE'] = application_test['AMT_INCOME_TOTAL'] - application_test['AMT_GOODS_PRICE']
for col in ['FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
           'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
            'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'WEEKDAY_APPR_PROCESS_START']:
    unique_values = list(set(list(application_train[col].astype(str).unique()) + list(application_test[col].astype(str).unique())))
    le.fit(unique_values)
    application_train[col] = le.transform(application_train[col].astype(str))
    application_test[col] = le.transform(application_test[col].astype(str))
train = application_train
train.head()
test = application_test
train = train.fillna(0)
test = test.fillna(0)
train.head()
X = train.drop(['SK_ID_CURR','contact_info', 'TARGET'], axis=1)
y = train['TARGET']
X_test = test.drop(['SK_ID_CURR'], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)
params = {
    'boosting': 'dart',
    'application': 'binary',
    'learning_rate': 0.01,
    'num_leaves': 34,
    'max_depth': 5,
    'feature_fraction': 0.9,
    'scale_pos_weight': 2,
    'reg_alpha': 0.05,
    'reg_lambda': 0.1}
model = lgb.train(params, lgb.Dataset(X_train, y_train), 
                  1000, 
                  [lgb.Dataset(X_train, y_train), lgb.Dataset(X_valid, y_valid)], 
                  verbose_eval=10, 
                  early_stopping_rounds=20)
lgb.plot_importance(model, max_num_features=100, figsize=(24, 18))
np.zeros(X_test.shape[0])
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
params={'colsample_bytree': 0.8,
 'learning_rate': 0.01,
 'num_leaves': 34,
 'subsample': 0.97,
 'max_depth': 8,
 'reg_alpha': 0.03,
 'reg_lambda': 0.07,
 'min_split_gain': 0.01,
 #'min_child_weight': 38
       }
prediction = np.zeros(X_test.shape[0])
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        train_x, train_y = X.iloc[train_idx], y.iloc[train_idx]
        valid_x, valid_y = X.iloc[valid_idx], y.iloc[valid_idx]
        clf = lgb.LGBMClassifier(**params)
        clf.fit(train_x, train_y, 
                eval_set = [(train_x, train_y), (valid_x, valid_y)], eval_metric = 'auc', 
                verbose = 100, early_stopping_rounds = 50)
        prediction += clf.predict(X_test)
sub = test[['SK_ID_CURR']].copy()
sub['TARGET'] = prediction / 10
sub.to_csv('Alim_sub.csv', index= False)
