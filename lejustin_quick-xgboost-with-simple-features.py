import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

plt.style.use('ggplot')

import seaborn as sns 



data = pd.read_csv('../input/aegypti_albopictus.csv')

print(data.head())
for c in data.columns:

    counts = data[c].value_counts()

    print("\nFrequency of each value of %s:\n" % c)

    print(counts.head())

    print("\nUnique values of %s: %i\n" % (c, len(counts)))
data = data.drop(['OCCURRENCE_ID', 'COUNTRY', 'COUNTRY_ID'], axis=1)



categoricals = ['VECTOR', 'SOURCE_TYPE', 'LOCATION_TYPE', 

                'POLYGON_ADMIN', 'GAUL_AD0', 'YEAR', 'STATUS']



for c in categoricals:

    data[c] = pd.factorize(np.array(data[c]))[0]



print(data.head())



sns.heatmap(data.corr())



plt.show()
grid = sns.FacetGrid(data, hue='VECTOR', row='LOCATION_TYPE')

grid.map(plt.scatter, 'X', 'Y')

grid.add_legend()



plt.show()
grid = sns.FacetGrid(data, hue='VECTOR', row='POLYGON_ADMIN')

grid.map(plt.scatter, 'X', 'Y')

grid.add_legend()



plt.show()
import xgboost as xgb 

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss, roc_auc_score, f1_score, accuracy_score



X = data.ix[:, 1:]

y = data.ix[:, 0]



# Instantiate XGBoost

n_estimators = 150

dtrain = xgb.DMatrix(X, y)



# XGBoost was tuned on the raw data.

bst = XGBClassifier(n_estimators=n_estimators,

                    max_depth=3, 

                    min_child_weight=5, 

                    gamma=0.5, 

                    learning_rate=0.05, 

                    subsample=0.7, 

                    colsample_bytree=0.7, 

                    reg_alpha=0.001,

                    seed=1,

                    silent=True)



# Cross-validate XGBoost

params = bst.get_xgb_params() # Extract parameters from XGB instance to be used for CV

num_boost_round = bst.get_params()['n_estimators'] # XGB-CV has different names than sklearn



cvresult = xgb.cv(params, dtrain, num_boost_round=num_boost_round, 

                  nfold=10, metrics=['logloss', 'auc', 'error'], seed=1)



# XGBoost summary

print("="*80)

print("\nXGBoost summary for 150 rounds of 10-fold cross-validation:")

print("\nBest mean log-loss: %.4f" % cvresult['test-logloss-mean'].min())

print("\nBest mean AUC: %.4f" % cvresult['test-auc-mean'].max())

print("\nBest mean error: %.4f" % cvresult['test-error-mean'].min())

print("="*80)



seed = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)



bst.fit(X_train, y_train, eval_metric='logloss')

pred = bst.predict(X_test)



print("="*80)

print("\nXGBoost performance on unseen data:")

print("\nlog-loss: %.4f" % log_loss(y_test, pred))

print("\nAUC: %.4f" % roc_auc_score(y_test, pred))

print("\nF1 score: %.4f" % f1_score(y_test, pred))

print("\nAccuracy: %.4f" % accuracy_score(y_test, pred))

print("="*80)
# Rename X temporarily to avoid confusion with the X variable

data = X



# -100 < X < -70     

# 20 < Y < 40                           

X_neg100_neg70 = np.where(data['X'] > -100, 1, 0) - np.where(data['X'] > -70, 1, 0)     

                                           

Y_20_40 = np.where(data['Y'] > 20, 1, 0) - np.where(data['Y'] > 40, 1, 0)     

                                      

data['X_neg100_neg70_Y_20_40'] = X_neg100_neg70*Y_20_40     

                                        

# -110 < X < -50               

# -40 < Y < 30                      

X_neg110_neg50 = np.where(data['X'] > -110, 1, 0) - np.where(data['X'] > -50, 1, 0)     

                                                                                          

Y_neg40_30 = np.where(data['Y'] > -40, 1, 0) - np.where(data['Y'] > 30, 1, 0)     

                                                                      

data['X_neg110_neg50_Y_neg40_30'] = X_neg110_neg50*Y_neg40_30



X = data

print(X.head())
dtrain = xgb.DMatrix(X, y)



cvresult = xgb.cv(params, dtrain, num_boost_round=num_boost_round, 

                  nfold=10, metrics=['logloss', 'auc', 'error'], seed=1)



# XGBoost summary

print("="*80)

print("\nXGBoost summary for 150 rounds of 10-fold cross-validation:")

print("\nBest mean log-loss: %.4f" % cvresult['test-logloss-mean'].min())

print("\nBest mean AUC: %.4f" % cvresult['test-auc-mean'].max())

print("\nBest mean error: %.4f" % cvresult['test-error-mean'].min())

print("="*80)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)



bst.fit(X_train, y_train, eval_metric='logloss')

pred = bst.predict(X_test)



print("="*80)

print("\nXGBoost performance on unseen data:")

print("\nlog-loss: %.4f" % log_loss(y_test, pred))

print("\nAUC: %.4f" % roc_auc_score(y_test, pred))

print("\nF1 score: %.4f" % f1_score(y_test, pred))

print("\nAccuracy: %.4f" % accuracy_score(y_test, pred))

print("="*80)