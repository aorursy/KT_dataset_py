import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(rc={'figure.figsize':(6,5)});
plt.figure(figsize=(6,5));

import os
print(os.listdir("../input"))
data = pd.read_csv('../input/PS_20174392719_1491204439457_log.csv')
print(data.shape)
data.head()
data = data[data.step <= 718]
print(data.shape)
# Data Cleaning
X = data[(data.type == 'TRANSFER') | (data.type == 'CASH_OUT')].copy()
#Y = X.isFraud

X.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis = 1, inplace = True)
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
X['errorBalanceOrig'] = X.newbalanceOrig + X.amount - X.oldbalanceOrg
X['errorBalanceDest'] = X.oldbalanceDest + X.amount - X.newbalanceDest
X.loc[X.type == 'TRANSFER', 'type'] = 0
X.loc[X.type == 'CASH_OUT', 'type'] = 1
X.type = X.type.astype(int) # convert dtype('O') to dtype(int)
X_train = X[X.step <= 600]
X_test = X[X.step > 600]
print(X_train.shape)
print(X_test.shape)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost.sklearn import XGBClassifier
rf = RandomForestClassifier(n_estimators=50, max_depth=3)
rf.fit(X_train.drop('isFraud', axis=1), X_train.isFraud)
rf_pred = rf.predict_proba(X_test.drop('isFraud', axis=1))[:, 1]
print('AUPRC = {}'.format(average_precision_score(X_test.isFraud, rf_pred)))
weights = (X.isFraud == 0).sum() / (1.0 * (X.isFraud == 1).sum())
xgb = XGBClassifier(max_depth = 3, scale_pos_weight = weights, n_jobs = 4)
#clf = XGBClassifier(max_depth = 3, n_jobs = 4)
xgb.fit(X_train.drop('isFraud', axis=1), X_train.isFraud)
xgb_prob = xgb.predict_proba(X_test.drop('isFraud', axis=1))[:, 1]
xgb_pred = xgb.predict(X_test.drop('isFraud', axis=1))
print('AUPRC = {}'.format(average_precision_score(X_test.isFraud, xgb_prob)))
X_test.head()
X_test[X_test.isFraud == 1].head()
from xgboost import plot_importance, to_graphviz
fig = plt.figure(figsize = (14, 9))
ax = fig.add_subplot(111)

colours = plt.cm.Set1(np.linspace(0, 1, 9))

ax = plot_importance(xgb, height = 1, color = colours, grid = False, \
                     show_values = False, importance_type = 'cover', ax = ax);
for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        
ax.set_xlabel('importance score', size = 16);
ax.set_ylabel('features', size = 16);
ax.set_yticklabels(ax.get_yticklabels(), size = 12);
ax.set_title('Ordering of features by importance to the model learnt', size = 20);
from sklearn.metrics import confusion_matrix

confusion_matrix(X_test.isFraud, xgb_pred)
