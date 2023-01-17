import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('creditcard_train.csv')
test = pd.read_csv('creditcard_test.csv')
%matplotlib inline
from scipy.stats import skew
from scipy.stats import norm
from scipy.special import boxcox1p, inv_boxcox
from datetime import date, datetime
import time
print(train.isnull().sum().sum())
print(test.isnull().sum().sum())
print(train.isna().sum().sum())
print(test.isna().sum().sum())
test.isnull().values.any()
test.head(10)
sns.countplot(train['Class'])
train.groupby('Class').size()

plt.subplots(figsize=(12, 9))
sns.heatmap(train.corr(), square=True, annot=True, fmt='.2f')
sns.heatmap(train[['Time', 'Class']].corr(), fmt='.4f', annot=True, square=True)
train[train['Amount'] > 1000]['Class'].describe()
train['Amount'].describe()
sns.distplot(train['V10'])
sns.distplot(train['Amount'])
c = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
       'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
from sklearn import preprocessing
scaler = preprocessing.RobustScaler()
train_X = scaler.fit_transform(train[c])
train_X = pd.DataFrame(train_X, columns=c)
sns.distplot(train_X['Amount'])
train_X.columns

training_features = train_X
training_target = train['Class']
from sklearn.model_selection import train_test_split, cross_val_score
x_train, x_val, y_train, y_val = train_test_split(training_features, training_target,
                                                  test_size = .1,
                                                  random_state=12)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(RandomForestClassifier(), scoring="f1", cv=3, verbose=1,
                  param_grid={"n_estimators": [20, 50, 100, 150], 
                              "max_depth":[2, 3, 4, 5],
                             }, )
gs.fit(training_features, training_target)
gs.best_params_


from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators=150, max_depth=5,random_state=12)
clf_rf.fit(x_train, y_train)
from sklearn.metrics import recall_score, f1_score, precision_score
print ('Validation Results')
print (clf_rf.score(x_val, y_val))
print (recall_score(y_val, clf_rf.predict(x_val)))
print (f1_score(y_val, clf_rf.predict(x_val)))
print (precision_score(y_val, clf_rf.predict(x_val)))
import xgboost as xgb
model_xgb = xgb.XGBClassifier(colsample_bytree=0.903, gamma=0.048, learning_rate=0.05, max_depth=5, 
                             min_child_weight=0.7817, n_estimators=3000, reg_alpha=0.7640, reg_lambda=0.8571,
                             subsample=0.8213, silent=1, random_state =7, n_jobs = -1)

from sklearn.model_selection import KFold, cross_val_score
def rmse_cv(model):
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    f1 = cross_val_score(model, x_train, y_train, scoring="f1", cv = kf)  # 默认的cv没有shuffle
    return(f1.mean())
rmse_cv(model_xgb)
model_xgb.fit(x_train, y_train)
from sklearn.metrics import recall_score, f1_score, precision_score
print ('Validation Results')
print (model_xgb.score(x_val, y_val))
print (recall_score(y_val, model_xgb.predict(x_val)))
print (f1_score(y_val, model_xgb.predict(x_val)))
print (precision_score(y_val, model_xgb.predict(x_val)))
test_X = scaler.transform(test[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
       'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']])
test_X = pd.DataFrame(test_X, columns=c)
predict_y = model_xgb.predict(test_X)
predict_y
res = pd.DataFrame([test['Id'].values, predict_y], index=['Id', 'Class']).T
res.to_csv('res2.csv', index=False, header=True)