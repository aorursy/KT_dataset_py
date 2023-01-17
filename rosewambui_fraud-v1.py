import sys

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline



import xgboost as xgb

from imblearn.ensemble import BalancedRandomForestClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, make_scorer, matthews_corrcoef



import os

from sklearn.model_selection import GridSearchCV

from imblearn.over_sampling import SMOTE

from imblearn.combine import SMOTEENN

from imblearn.under_sampling import RandomUnderSampler

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

from collections import Counter

label = LabelEncoder()

print(os.listdir("../input"))



import warnings

warnings.filterwarnings("ignore")
np.set_printoptions(threshold=sys.maxsize)
train = pd.read_csv("../input/fraud-ug/training.csv")

train["train"] = 1

train.head()
train.shape
fraud_data = train[train["FraudResult"] == 1.0]
train.shape
test = pd.read_csv("../input/fraud-test/test.csv")

test["train"] = 0

test.head()
dataset = pd.concat([train, test], ignore_index=True)
dataset.shape
train.head()
# time

dataset["TransactionStartTime"] = dataset["TransactionStartTime"].apply(lambda x : pd.to_datetime(x))

# dataset["Month"] = dataset["TransactionStartTime"].dt.month

# dataset["day_of_month"] = dataset["TransactionStartTime"].dt.day

dataset["day_of_week"] = dataset["TransactionStartTime"].dt.dayofweek

# dataset["day_of_year"] = dataset["TransactionStartTime"].dt.dayofyear

dataset["time"] = dataset["TransactionStartTime"].dt.time

dataset["minute"] = dataset["time"].apply(lambda x: int(str(x).split(":")[0]) * 60 + int(str(x).split(":")[1]))
dataset.Amount.mean()
dataset.SubscriptionId = pd.DataFrame(label.fit_transform(dataset.SubscriptionId))

dataset.BatchId = pd.DataFrame(label.fit_transform(dataset.BatchId))

dataset.CustomerId = pd.DataFrame(label.fit_transform(dataset.CustomerId))

dataset.AccountId = pd.DataFrame(label.fit_transform(dataset.AccountId))

dataset = pd.concat([dataset, pd.get_dummies(dataset["ChannelId"], prefix="Channel_Id_")], axis=1)

dataset = pd.concat([dataset, pd.get_dummies(dataset["PricingStrategy"], prefix="PricingStrategy_")], axis=1)

dataset = pd.concat([dataset, pd.get_dummies(dataset["ProductCategory"], prefix="ProductCategory_")], axis=1)

dataset = pd.concat([dataset, pd.get_dummies(dataset["ProductId"], prefix="ProductId_")], axis=1)

dataset = pd.concat([dataset, pd.get_dummies(dataset["ProviderId"], prefix="ProviderId_")], axis=1)

dataset["weekday"] = dataset["TransactionStartTime"].dt.weekday_name
dataset = pd.concat([dataset, pd.get_dummies(dataset["weekday"], prefix="weekday")], axis=1)
# id

# dataset.SubscriptionId = pd.DataFrame(label.fit_transform(dataset.SubscriptionId))

# # dataset.BatchId = pd.DataFrame(label.fit_transform(dataset.BatchId))

# dataset.CustomerId = pd.DataFrame(label.fit_transform(dataset.CustomerId))

# dataset.AccountId = pd.DataFrame(label.fit_transform(dataset.AccountId))
# dataset["AccountId"] = dataset["AccountId"].apply(lambda x: int(x.split("_")[1]))

# dataset["SubscriptionId"] = dataset["SubscriptionId"].apply(lambda x: int(x.split("_")[1]))

# dataset["CustomerId"] = dataset["CustomerId"].apply(lambda x: int(x.split("_")[1]))

# dataset["BatchId"] = dataset["BatchId"].apply(lambda x: int(x.split("_")[1]))
dataset.shape
dataset.shape
# group = dataset[["AccountId", 'Amount', 'Month']].groupby(by=["AccountId", 'Amount'])[['Month']].mean().reset_index().rename(index=str, columns={'Month': 'Cust_value_month'})

# dataset = dataset.merge(group, how='left')
# group = dataset[["AccountId", 'Amount', 'Month']].groupby(by=["AccountId", 'Amount'])[['Month']].count().reset_index().rename(index=str, columns={'Month': 'Cust_value_month_count'})

# dataset = dataset.merge(group, how='left')
# group = dataset[['AccountId','Amount', 'PricingStrategy', 'ProductId','ProviderId', 'CustomerId', 'day_of_week', 'ChannelId']].groupby(by=['CustomerId', 'ProductId', 'ProviderId', 'ChannelId', 'Amount'])[['day_of_week']].mean().reset_index().rename(index=str, columns={'day_of_week': 'Cust_prod_mean_dayofweek'})

# dataset = dataset.merge(group, on=['CustomerId', 'ProductId', 'ProviderId', 'ChannelId'], how='left')

# group = dataset[['CustomerId', 'ProductId', 'ProviderId', 'ChannelId', 'Amount', 'day_of_week']].groupby(by=['CustomerId', 'ProductId', 'ProviderId', 'ChannelId', 'Amount',])[['day_of_week']].count().reset_index().rename(index=str, columns={'day_of_week': 'Cust_prod_mean_dayofweek'})

# dataset = dataset.merge(group, on=['CustomerId', 'ProductId', 'ProviderId', 'ChannelId',], how='left')
dataset.drop(["CurrencyCode", "CountryCode", "BatchId", "time", "TransactionStartTime", "ChannelId", "PricingStrategy", "ProductCategory", "ProductId", "ProviderId", "weekday"], axis=1, inplace=True)
train = dataset[dataset["train"] == 1]

test = dataset[dataset["train"] == 0]
test.shape
train.drop(["train"], axis=1, inplace=True)

test.drop(["train", "FraudResult"], axis=1, inplace=True)
from sklearn.decomposition import PCA





reduced = PCA(n_components=2).fit_transform(train.drop(["TransactionId", "FraudResult"], axis=1).values)



plt.figure()

plt.scatter(reduced[train["FraudResult"] == 0, 0], reduced[train["FraudResult"] == 0, 1], color='blue')

plt.scatter(reduced[train["FraudResult"] == 1, 0], reduced[train["FraudResult"] == 1, 1], color='red', marker='x')

plt.xlabel("PC1")

plt.ylabel("PC2")
from sklearn.model_selection import train_test_split
train.drop(["TransactionId"], axis=1, inplace=True)
X = train.drop(["FraudResult"], axis=1)

y = train["FraudResult"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print("Fraud case", (train["FraudResult"].value_counts()[1]/ len(train) * 100), "% of the dataset")

print ("Train size", y_train.shape[0], "Fraud in train size", y_train[y_train == 1].shape[0] / y_train.shape[0] * 100)

print ("Test size", y_test.shape[0], "Fraud in tesst size", y_test[y_test == 1].shape[0] / y_test.shape[0] * 100)
# # implement grid search

# rfc_param_grid = {

#     'n_estimators': [200, 250, 300, 400],

#     'max_depth': [5, 7, 8, 10],

#   

# }

# rfc_sk =  RandomForestClassifier(random_state=1)
# from sklearn.model_selection import GridSearchCV

# grid_mse = GridSearchCV(param_grid=rfc_param_grid, estimator=rfc_sk,

# scoring="f1", cv=4 )
# grid_mse.fit(X, y)
# grid_mse.best_params_
# grid_mse.best_score_
from sklearn.svm import SVC
svc = SVC(random_state=1)
gbc = GradientBoostingClassifier(n_estimators=200, max_depth=5)
gbc.fit(X_train, y_train)
gbc_pred = gbc.predict(X_test)
print(classification_report(y_test, gbc_pred))
# classifier

xg_cl = xgb.XGBClassifier(objective="binary:logistic",

                          n_estimators=200, colsample_bytree=0.7, subsample=0.7, max_depth=5, learning_rate=0.1, seed=1)

MCC_scorer = make_scorer(matthews_corrcoef)
# rfc = RandomForestClassifier(random_state=1, max_depth=8, n_estimators=200,)

pipeline_rf = Pipeline([

    ('model', RandomForestClassifier(max_depth=9, n_jobs=-1, random_state=1))

])

param_grid_rf = {'model__n_estimators': [50, 75, 100, 150, 200, 250, 300]

                 }



grid_rf = GridSearchCV(estimator=pipeline_rf, param_grid=param_grid_rf,

                       scoring=MCC_scorer, n_jobs=-1, pre_dispatch='2*n_jobs',

                       cv=8, verbose=1, return_train_score=False)



grid_rf.fit(X_train, y_train)
grid_rf.fit(X_train, y_train)
grid_rf_pred = grid_rf.predict(X_test)
print("rfc ", classification_report(y_test, grid_rf_pred) )
xg_cl.fit(X_train, y_train)
preds = xg_cl.predict(X_test)
print(classification_report(y_test, preds))
test.head()
xgb.plot_importance(xg_cl)

plt.rcParams['figure.figsize'] = [15, 5]
# combine models

# gbc

# xgb

# rfc

models = [

    xgb.XGBClassifier(objective="binary:logistic",

                          n_estimators=200, colsample_bytree=0.7, subsample=0.7, max_depth=5, learning_rate=0.1, seed=1),

    RandomForestClassifier(random_state=1, max_depth=8, n_estimators=200,)

    

]



preds = pd.DataFrame()

for i, m in enumerate(models):

    print(m)

    m.fit(X_train, y_train),

    preds[i] = m.predict_proba(X_test)[:,1]



weights = [0.3, 1]

preds['weighted_pred'] = (preds * weights).sum(axis=1) / sum(weights)

preds.head()
total_pred = np.where(preds["weighted_pred"] > 0.5, 1, 0)
print(classification_report(y_test, total_pred))
test_pred = pd.DataFrame()

test_pred['TransactionId'] = test["TransactionId"]

test_sub = test.copy()

test_sub.drop(["TransactionId"], axis=1, inplace=True)
models = [

    xgb.XGBClassifier(objective="binary:logistic",

                          n_estimators=200, colsample_bytree=0.7, subsample=0.7, max_depth=5, learning_rate=0.1, seed=1),

    RandomForestClassifier(random_state=1, max_depth=8, n_estimators=200,)

    

]



preds = pd.DataFrame()

for i, m in enumerate(models):

    m.fit(X_train, y_train),

    preds[i] = m.predict_proba(test_sub)[:,1]



weights = [0.3, 1]

preds['weighted_pred'] = (preds * weights).sum(axis=1) / sum(weights)

preds.head()
# predict_test = np.where(preds["weighted_pred"] > 0.5, 1, 0)

predict_test = grid_rf.predict(test_sub)

test_pred['FraudResult'] = predict_test

test_pred.to_csv('submission.csv', index=False)
import lightgbm as lgbm
X_train, X_valid, Y_train, Y_valid = train_test_split(X, y, test_size=0.33, random_state=42)
# making lgbm datasets for train and valid

d_train = lgbm.Dataset(X_train, Y_train)

d_valid = lgbm.Dataset(X_valid, Y_valid)

    
params = {

    'objective' :'binary',

    'learning_rate' : 0.02,

    'num_leaves' : 76,

    'feature_fraction': 0.64, 

    'bagging_fraction': 0.8, 

    'bagging_freq':1,

    'boosting_type' : 'gbdt',

    'metric': 'binary_logloss'

}
bst =lgbm.train(params,

                d_train,

                valid_sets=[d_valid],

                verbose_eval=10,

                early_stopping_rounds=100)

    
test.head()
test_pred = pd.DataFrame()

test_pred['TransactionId'] = test["TransactionId"]

test.drop(["TransactionId"], axis=1, inplace=True)
lgbm_preds = bst.predict(test)

len(lgbm_preds)

# lgbm_preds = np.where(lgbm_preds > 0.5, 1, 0)

len(lgbm_preds)

test_pred['FraudResult'] = lgbm_preds

test_pred.to_csv('submission_lgbm.csv', index=False)
test.shape
# len(y_pred)
# print('this SVM', classification_report(y_test, y_pred_test))