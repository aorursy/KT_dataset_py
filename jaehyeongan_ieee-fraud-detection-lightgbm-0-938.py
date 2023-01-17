import pandas as pd

import numpy as np

import os

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score, recall_score, classification_report

import lightgbm as lgb



# path

path_dir = '../input/ieee-fraud-detection/'

file_list = os.listdir(path_dir)

file_list
# Train Data

train_identity = pd.read_csv(path_dir+'train_identity.csv')

train_transaction = pd.read_csv(path_dir+'train_transaction.csv')

# Test Data

test_identity = pd.read_csv(path_dir+'test_identity.csv')

test_transaction = pd.read_csv(path_dir+'test_transaction.csv')
train_identity.head()
train_transaction.head()
train_merge = pd.merge(train_identity, train_transaction, on=['TransactionID'], how='right')

test_merge = pd.merge(test_identity, test_transaction, on=['TransactionID'], how='right')

train_merge = train_merge.drop(['DeviceInfo'], axis=1)

test_merge = test_merge.drop(['DeviceInfo'], axis=1)



del train_identity, train_transaction, test_identity, test_transaction
train_merge.head()
# Check NULL ratio >= 0.77

check_null = train_merge.isna().sum() / len(train_merge)

check_null[check_null >= 0.77]
# remove cols of null ratio >= 0.77

remove_cols = list(check_null[check_null >= 0.77].keys())

not_remove_cols = ['id_30','id_31','R_emaildomain','P_emaildomain','ProductCD','card4','card6','DeviceType']



for col in not_remove_cols:

    if col in remove_cols:

        remove_cols.remove(col)

    else:

        pass



train_merge = train_merge.drop(remove_cols, axis=1)

test_merge = test_merge.drop(remove_cols, axis=1)

print(train_merge.shape)

print(test_merge.shape)
train_merge.head()
# select types

object_cols = train_merge.select_dtypes(include='object').columns

nobject_cols = train_merge.select_dtypes(exclude='object').columns



print('Columns of Object Type\n{}'.format(object_cols.values))
train_merge[object_cols] = train_merge[object_cols].fillna('NaN')

train_merge[nobject_cols] = train_merge[nobject_cols].fillna(0)



test_merge[object_cols] = test_merge[object_cols].fillna('NaN')

nobject_cols = list(nobject_cols)

nobject_cols.remove('isFraud')

test_merge[nobject_cols] = test_merge[nobject_cols].fillna(0)
train_merge.head()
# Ratio of isFraud(target value)

isfraud_values = train_merge['isFraud'].value_counts().values

isfraud_0 = (isfraud_values[0] / len(train_merge['isFraud'])) * 100

isfraud_1 = (isfraud_values[1] / len(train_merge['isFraud'])) * 100

print('Ratio of isFraud 0 : {:.1f} %'.format(isfraud_0))

print('Ratio of isFraud 1 : {:.1f} %'.format(isfraud_1))



# plot

plt.figure(figsize=(7, 4))

sns.countplot(x='isFraud', data=train_merge)

plt.xlabel('isFraud'); plt.ylabel('Count')

plt.title('Count of Target Value'); plt.show()
cols = ['DeviceType', 'ProductCD', 'card4', 'card6']

plt.figure(figsize=(11, 15))

for i, col in enumerate(cols):

    plt.subplot(len(cols), 1, i+1)

    sns.countplot(x=col,  hue='isFraud', data=train_merge)

    plt.title('Count by '+str(col))

    plt.tight_layout()
# Distribution of TransactionAmt by isFraud

plt.figure(figsize=(11, 8))

for i in [0, 1]:

    plt.subplot(2,1, i+1)

    sns.distplot(train_merge[train_merge['isFraud'] == i][train_merge['TransactionAmt'] <=700]['TransactionAmt'])

    plt.title('Distribution of TransactionAmt by isFraud '+str(i))

    plt.tight_layout()
# Distribution of TransactionAmt by DeviceType

devicetype = train_merge['DeviceType'].unique()

plt.figure(figsize=(11, 8))

for i, device in enumerate(devicetype):

    plt.subplot(len(devicetype), 1, i+1)

    sns.distplot(train_merge[train_merge['DeviceType'] == device][train_merge['TransactionAmt'] <=700]['TransactionAmt'])

    plt.title('Distribution of TransactionAmt by '+str(device))

    plt.tight_layout()
productcd = train_merge['ProductCD'].unique()

plt.figure(figsize=(11, 9))

for i, cd in enumerate(productcd):

    plt.subplot(len(productcd), 1, i+1)

    sns.distplot(train_merge[train_merge['ProductCD']==cd][train_merge['TransactionAmt'] <=700]['TransactionAmt'])

    plt.title('Distribution of TransactionAmt by '+str(cd))

    plt.tight_layout()
# unique한 특성이 너무 많아 데이터 단순화 필요 

print('id_30 : {}'.format(train_merge['id_30'].unique())+'\n')

print('id_31 : {}'.format(train_merge['id_31'].unique())+'\n')

print('P_emaildomain : {}'.format(train_merge['P_emaildomain'].unique())+'\n')

print('R_emaildomain : {}'.format(train_merge['R_emaildomain'].unique())+'\n')
def simplify_categorical(data):

    mobile_os = [i.split(' ')[0] for i in data['id_30']]

    browser = []

    for i in data['id_31']:

        if i.split(' ')[0] == 'mobile' and len(i.split(' ')) > 1: # ex. mobile safari 10.0, mobile safari generic, ...

            browser.append(i.split(' ')[1])

        elif i.split('/')[0] == 'Samsung':    # ex. Samsung/SM-G532M, Samsung/SCH, Samsung/SM-G531H,

            browser.append('samsung')

        elif i.split('/')[0] == 'Microsoft':  # ex. Microsoft/Windows

            browser.append('ie')

        elif i.split('/')[0] == 'Mozilla':    # ex. Mozilla/Firefox

            browser.append('firefox')

        elif i.split('/')[0] == 'Generic':    # ex. Generic/Android

            browser.append('android')

        elif len(i.split(' ')) >= 1:          # ex. edge 14.0, android brower 4.0, Chrome 63.0 for Android, ...

            browser.append(i.split(' ')[0])

        else:

            browser.append(i)                  # android, chrome, edge, google, le, ...

    p_emaildomain = [i.split('.')[0] for i in data['P_emaildomain']]

    r_emaildomain = [i.split('.')[0] for i in data['R_emaildomain']]

    

    data['id_30'] = [i.lower() for i in mobile_os]

    data['id_31'] = [i.lower() for i in browser]

    data['P_emaildomain'] = [i.lower() for i in p_emaildomain]

    data['R_emaildomain'] = [i.lower() for i in r_emaildomain]

    

    return data



train = simplify_categorical(train_merge)

train_target = train['isFraud']

train = train.drop('isFraud', axis=1)

test = simplify_categorical(test_merge)



del train_merge, test_merge
print('id_30 : {}'.format(train['id_30'].unique())+'\n')

print('id_31 : {}'.format(train['id_31'].unique())+'\n')

print('P_emaildomain : {}'.format(train['P_emaildomain'].unique())+'\n')

print('R_emaildomain : {}'.format(train['R_emaildomain'].unique())+'\n')
# LabelEncoding 

categorical_features = list(object_cols)



for col in categorical_features:

    le = LabelEncoder()

    le.fit(list(train[col].values) + list(test[col].values))

    train[col] = le.transform(list(train[col].values))

    test[col] = le.transform(list(test[col].values))
train_corr = train[:50000].copy()

train_corr['isFraud'] = train_target.copy()



corrmat = train_corr.corr()

top_corr_features = corrmat.index[abs(corrmat['isFraud']) >= 0.10]

# top_corr_features

plt.figure(figsize=(13,10))

sns.heatmap(train_corr[top_corr_features].corr(), annot=False, cmap="RdYlGn")

plt.title('Variable Correlations')

plt.show()



del corrmat
train = train.drop('TransactionID', axis=1)

test_tid = test['TransactionID']

test = test.drop('TransactionID', axis=1)
# Standardization

train_scale_ = train.drop(categorical_features, axis=1)

test_scale_ = test.drop(categorical_features, axis=1)



# z = (x - u) / s

train_scale = (train_scale_ - train_scale_.mean()) / train_scale_.std()

test_scale = (test_scale_ - train_scale_.mean()) / train_scale_.std()
train[train_scale.columns] = train_scale

test[test_scale.columns] = test_scale



del train_scale_, test_scale_, train_scale, test_scale
pca = PCA(n_components=3)

pca.fit(train)

pca_train = pca.transform(train)

pca_test = pca.transform(test)
train['pca_col1'] = pca_train[:, 0]

train['pca_col2'] = pca_train[:, 1]

train['pca_col3'] = pca_train[:, 2]



test['pca_col1'] = pca_test[:, 0]

test['pca_col2'] = pca_test[:, 1]

test['pca_col3'] = pca_test[:, 2]



del pca_train, pca_test
train.columns
# Split train set / valid set

x_train, x_val, y_train, y_val = train_test_split(train, train_target, test_size=0.3, random_state=42)
# LightGBM

lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=categorical_features)

lgb_val = lgb.Dataset(x_val, y_val, categorical_feature=categorical_features)



# parameters

params = {

    'objective':'binary',

    'boosting_type': 'gbdt',

    'max_depth': -1,

    'learning_rate': 0.03,

    'num_leaves': 500,        # number of leaves in full tree

    'feature_fraction':0.7,   # selected parameters ratio in each iteration for building trees

    'bagging_fraction':0.8,   # specifies the fraction of data to be used for each iteration

    'bagging_seed':11,        # random seed for bagging

    'n_jobs':-1,

    'verbosity': -1

}



# Training

lgb_model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_val],

                     num_boost_round=3000,

                     early_stopping_rounds=500,

                     verbose_eval=100)
val_preds_lgb = [1 if i>=0.5 else 0 for i in lgb_model.predict(x_val)]



print('Accuracy : {:.2f}'.format(accuracy_score(val_preds_lgb, y_val)))

print('Recall : {:.2f}'.format(recall_score(val_preds_lgb, y_val)))

print(classification_report(val_preds_lgb, y_val))
# feature importances

feature_importance = lgb_model.feature_importance()

df_fi = pd.DataFrame({'columns':x_train.columns, 'importances':feature_importance})

df_fi = df_fi[df_fi['importances'] > 500].sort_values(by=['importances'], ascending=False)



fig = plt.figure(figsize=(15, 6))

ax = sns.barplot(df_fi['columns'], df_fi['importances'])

ax.set_xticklabels(df_fi['columns'], rotation=80, fontsize=13)

plt.title('Feature Importance')

plt.tight_layout()

plt.show()
test_preds = lgb_model.predict(test)
test_pred_df = pd.DataFrame()

test_pred_df['TransactionID'] = test_tid

test_pred_df['isFraud'] = test_preds
test_pred_df.head(10)
test_pred_df.to_csv('lgb_submission.csv', index=False)