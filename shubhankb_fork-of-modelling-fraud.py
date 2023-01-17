import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(rc={'figure.figsize':(6,5)});
plt.figure(figsize=(6,5));

import os
print(os.listdir("../input"))
data = pd.read_csv('../input/paysim1/PS_20174392719_1491204439457_log.csv')
print(data.shape)
data.head()
base_cols = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
data = data[data.step <= 718]
print(data.shape)
data[(data.amount > 1) & (data.amount < 1000000) & (data.isFraud == 0)].amount.hist(bins=30)
data['Hour_of_day'] = data.step % 24
data['Day_of_month'] = np.ceil(data.step / 24)
data['Day_of_week'] = (data.Day_of_month % 7) + 1
#data.drop('Day_of_month', axis=1, inplace=True)
data.head()
temp = data.groupby(['Day_of_month'])[['type']].count()
temp.reset_index(inplace=True)
temp.type.plot.bar()
# Frauster have a habit of not targeting very high and very low amount
print(data[data.isFraud==1].amount.describe())
data['Fraud_Habit_flag'] = ((data.amount > 125000)&(data.amount < 1525000)).astype(int)
print(data.groupby(['isFraud']).oldbalanceOrg.describe())
# Average ... high balance account
print(data[data.isFraud==1].oldbalanceOrg.describe())
data['is_origin_openBal_high'] = (data.oldbalanceOrg > 125000).astype(int)
# Average ... high balance account
# Opening balance high and closing balance is zero
data['origin_anamoly_flag'] = ((data.oldbalanceOrg > 125000)&(data.newbalanceOrig < 1)).astype(int)
temp = data[data.step < 600].groupby(['type'])[['amount']].agg(['mean', 'std']).reset_index()
temp.columns = ['type', 'type_amount_mean', 'type_amount_std']
data = data.merge(temp, how='left')
data.head()
#temp = data[data.step < 600].groupby(['type', 'isFraud'])[['amount']].agg(['mean', 'std']).reset_index()
#temp.columns = ['type', 'isFraud', 'type_amount_mean', 'type_amount_std']
#data = data.merge(temp, how='left')
#data.head()
'''
temp = data.groupby('nameDest')[['type']].count()
temp.reset_index(inplace=True)

temp1 = data[data.isFraud==1].groupby('nameDest')[['step']].count()
temp1.reset_index(inplace=True)

temp = temp.merge(temp1, how='left')
temp.fillna(0, inplace=True)
temp['Dest_fraud_ratio'] = temp['step']/temp.type

data = data.merge(temp[['nameDest', 'Dest_fraud_ratio']], how='left')
del temp, temp1
data.head()
'''
# Data Cleaning
X = data[(data.type == 'TRANSFER') | (data.type == 'CASH_OUT')].copy()
#Y = X.isFraud

X.drop(['nameOrig', 'nameDest', 'isFlaggedFraud', 'Day_of_month'], axis = 1, inplace = True)
print(X.shape)
#print(Y.shape)
X.head()
X.isFraud.value_counts(normalize=True)*100
X[X.step > 600].shape
X[X.step > 600].isFraud.value_counts(normalize=True)
# Possible Missing Values encoded as zeros.
#X.loc[(X.oldbalanceDest == 0) & (X.newbalanceDest == 0) & (X.amount != 0), \
#      ['oldbalanceDest', 'newbalanceDest']] = - 1
#X.loc[(X.oldbalanceOrg == 0) & (X.newbalanceOrig == 0) & (X.amount != 0), \
#      ['oldbalanceOrg', 'newbalanceOrig']] = np.nan
#X.drop(['errorBalanceOrig', 'errorBalanceDest'], axis=1, inplace=True)
#X['errorBalanceOrig'] = X.newbalanceOrig + X.amount - X.oldbalanceOrg
#X['errorBalanceDest'] = X.oldbalanceDest + X.amount - X.newbalanceDest
X.Hour_of_day = X.Hour_of_day.astype(int)
X.Day_of_week = X.Day_of_week.astype(int)
X.head()
X.loc[X.type == 'TRANSFER', 'type'] = 0
X.loc[X.type == 'CASH_OUT', 'type'] = 1
X.type = X.type.astype(int) # convert dtype('O') to dtype(int)
X_train = X[X.step <= 600]
X_test = X[X.step > 600]
X_train.drop('step', axis=1, inplace=True)
X_test.drop('step', axis=1, inplace=True)
print(X_train.shape)
print(X_test.shape)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train[base_cols], X_train.isFraud)
lr_pred_train = lr.predict_proba(X_train[base_cols])[:, 1]
lr_pred = lr.predict_proba(X_test[base_cols])[:, 1]
#print('Train AUPRC = {}'.format(average_precision_score(X_train.isFraud, lr_pred_train)))
print('Test AUPRC = {}'.format(average_precision_score(X_test.isFraud, lr_pred)))
rf = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=4)
rf.fit(X_train[base_cols], X_train.isFraud)
rf_pred_train = rf.predict_proba(X_train[base_cols])[:, 1]
rf_pred = rf.predict_proba(X_test[base_cols])[:, 1]
#print('Train AUPRC = {}'.format(average_precision_score(X_train.isFraud, rf_pred_train)))
print('Test AUPRC = {}'.format(average_precision_score(X_test.isFraud, rf_pred)))
X_train.columns
feat_cols = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
       'newbalanceDest', 'Hour_of_day', 'Day_of_week', 'type_amount_mean', 'type_amount_std',
       'Fraud_Habit_flag', 'is_origin_openBal_high', 'origin_anamoly_flag']
lr = LogisticRegression()
lr.fit(X_train[feat_cols], X_train.isFraud)
lr_pred_train = lr.predict_proba(X_train[feat_cols])[:, 1]
lr_pred = lr.predict_proba(X_test[feat_cols])[:, 1]
#print('Train AUPRC = {}'.format(average_precision_score(X_train.isFraud, lr_pred_train)))
print('Test AUPRC = {}'.format(average_precision_score(X_test.isFraud, lr_pred)))
rf = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=4)
rf.fit(X_train[feat_cols], X_train.isFraud)
rf_pred_train = rf.predict_proba(X_train[feat_cols])[:, 1]
rf_pred = rf.predict_proba(X_test[feat_cols])[:, 1]
#print('Train AUPRC = {}'.format(average_precision_score(X_train.isFraud, rf_pred_train)))
print('Test AUPRC = {}'.format(average_precision_score(X_test.isFraud, rf_pred)))
weights = (X.isFraud == 0).sum() / (1.0 * (X.isFraud == 1).sum())
xgb = XGBClassifier(max_depth = 5, scale_pos_weight = weights, n_jobs = 4)
#clf = XGBClassifier(max_depth = 3, n_jobs = 4)
xgb.fit(X_train[feat_cols], X_train.isFraud)
xgb_pred_train = xgb.predict_proba(X_train[feat_cols])[:, 1]
xgb_pred = xgb.predict_proba(X_test[feat_cols])[:, 1]
#print('Train AUPRC = {}'.format(average_precision_score(X_train.isFraud, xgb_pred_train)))
print('Test AUPRC = {}'.format(average_precision_score(X_test.isFraud, xgb_pred)))
xgb_prob = xgb.predict_proba(X_test.drop('isFraud', axis=1))[:, 1]
xgb_pred = xgb.predict(X_test.drop('isFraud', axis=1))
print('AUPRC = {}'.format(average_precision_score(X_test.isFraud, xgb_prob)))
from xgboost import plot_importance, to_graphviz
fig = plt.figure(figsize = (14, 9))
ax = fig.add_subplot(111)

#colours = plt.cm.Set1(np.linspace(0, 1, 9))

ax = plot_importance(xgb, height = 1, grid = False, \
                     show_values = False, importance_type = 'cover', ax = ax);
for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        
ax.set_xlabel('importance score', size = 16);
ax.set_ylabel('features', size = 16);
ax.set_yticklabels(ax.get_yticklabels(), size = 12);
ax.set_title('Ordering of features by importance to the model learnt', size = 20);
from sklearn.metrics import confusion_matrix

confusion_matrix(X_test.isFraud, xgb_pred)
feat_cols = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
       'newbalanceDest', 'Hour_of_day', 'Day_of_week',
       'Fraud_Habit_flag', 'is_origin_openBal_high', 'origin_anamoly_flag']
balanced_data = pd.read_csv('../input/fraud-analysis-in-r/balanced_data.csv')
print(balanced_data.shape)
balanced_data.head()
test_data = pd.read_csv('../input/fraud-analysis-in-r/test_data.csv')
test_data.head()
balanced_data.drop(['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True)
balanced_data.loc[balanced_data.type == 'TRANSFER', 'type'] = 0
balanced_data.loc[balanced_data.type == 'CASH_OUT', 'type'] = 1
balanced_data.type = balanced_data.type.astype(int) # convert dtype('O') to dtype(int)

test_data.loc[test_data.type == 'TRANSFER', 'type'] = 0
test_data.loc[test_data.type == 'CASH_OUT', 'type'] = 1
test_data.type = test_data.type.astype(int) # convert dtype('O') to dtype(int)
lr = LogisticRegression()
lr.fit(balanced_data[feat_cols], balanced_data.isFraud)
lr_pred = lr.predict_proba(test_data[feat_cols])[:, 1]
print('AUPRC = {}'.format(average_precision_score(test_data.isFraud, lr_pred)))
rf = RandomForestClassifier(n_estimators=100, max_depth=5)
rf.fit(balanced_data[feat_cols], balanced_data.isFraud)
rf_pred = rf.predict_proba(test_data[feat_cols])[:, 1]
print('AUPRC = {}'.format(average_precision_score(test_data.isFraud, rf_pred)))
importances = rf.feature_importances_
indices = np.argsort(importances)

plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), balanced_data.columns[indices])
plt.xlabel('Relative Importance')
xgb = XGBClassifier(max_depth = 3, n_jobs = 4)
xgb.fit(balanced_data[feat_cols], balanced_data.isFraud)
xgb_prob = xgb.predict_proba(test_data[feat_cols])[:, 1]
xgb_pred = xgb.predict(test_data[feat_cols])
print('AUPRC = {}'.format(average_precision_score(test_data.isFraud, xgb_pred)))
from xgboost import plot_importance, to_graphviz
fig = plt.figure(figsize = (14, 9))
ax = fig.add_subplot(111)

ax = plot_importance(xgb, height = 1, grid = False, show_values = False, importance_type = 'cover', ax = ax);
for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        
ax.set_xlabel('importance score', size = 16);
ax.set_ylabel('features', size = 16);
ax.set_yticklabels(ax.get_yticklabels(), size = 12);
ax.set_title('Ordering of features by importance to the model learnt', size = 20);
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(balanced_data.drop('isFraud', axis=1), balanced_data.isFraud,
                                                    test_size=0.2, stratify = balanced_data.isFraud, random_state=42)

print(X_train.shape)
print(X_test.shape)
y_test.value_counts(normalize=True)
rf = RandomForestClassifier(n_estimators=50, max_depth=3)
rf.fit(X_train[feat_cols], y_train)
rf_pred = rf.predict_proba(X_test[feat_cols])[:, 1]
print('AUPRC = {}'.format(average_precision_score(y_test.isFraud, rf_pred)))
importances = rf.feature_importances_
indices = np.argsort(importances)

plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), X_train[feat_cols].columns[indices])
plt.xlabel('Relative Importance')
xgb = XGBClassifier(max_depth = 3, n_jobs = 4)
xgb.fit(X_train[feat_cols], y_train)
xgb_prob = xgb.predict_proba(X_train[feat_cols])[:, 1]
xgb_pred = xgb.predict(X_train[feat_cols])
print('AUPRC = {}'.format(average_precision_score(y_test, xgb_pred)))
from xgboost import plot_importance, to_graphviz
fig = plt.figure(figsize = (14, 9))
ax = fig.add_subplot(111)

ax = plot_importance(xgb, height = 1, grid = False, show_values = False, importance_type = 'cover', ax = ax);
for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        
ax.set_xlabel('importance score', size = 16);
ax.set_ylabel('features', size = 16);
ax.set_yticklabels(ax.get_yticklabels(), size = 12);
ax.set_title('Ordering of features by importance to the model learnt', size = 20);
