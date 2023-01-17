import numpy as np
import pandas as pd 
from scipy import stats
from statsmodels.stats.descriptivestats import sign_test
from statsmodels.stats.weightstats import zconfint
from statsmodels.stats.weightstats import *
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
import sklearn
import scipy.sparse 
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline 

pd.set_option('display.max_rows', 600)
pd.set_option('display.max_columns', 50)
sns.set(rc={'figure.figsize':(20, 10)})
#data download
train = pd.read_csv("../input/train-data/traindata.csv")
test = pd.read_csv('../input/test-data/testdata.csv')
#sample_subm = pd.read_csv('sample_submission.csv')
print(train.shape)
train.head()
print(test.shape)
test.head()
train.describe()
test.describe()
train.dtypes
train['y'].value_counts()
preds = list(set(train.columns)-set(['Unnamed: 0', 'y']))
len(preds)
sns.pairplot(train)
#categorical columns
cat_cols = list(train.select_dtypes(include='O').columns)
#dummy encoding
train = train.join(pd.get_dummies(train[cat_cols]))
test = test.join(pd.get_dummies(test[cat_cols]))

print (train.shape)
print (test.shape)
#mean encoding
cat_cols.append('day')
cat_cols.append('campaign')

# 1 mean encoding
y_train = train['y']
maps_for_cat_means = dict()
maps_for_cat_min = dict()
maps_for_cat_max = dict()
for field in cat_cols:
    print(field)
    target_mean = train.groupby(field).y.mean()
    maps_for_cat_means[field + '_target_enc'] = target_mean    
    
for field in cat_cols:
    train[field + '_enc_mean'] = train[field].map(maps_for_cat_means[field + '_target_enc'])
    test[field + '_enc_mean'] = test[field].map(maps_for_cat_means[field + '_target_enc'])
print (train.shape)
print (test.shape)
train.head()
#some new features
fields_for_prep = ['age', 'balance', 'balance', 'day', 'duration']

for field in fields_for_prep:
    mean = np.mean(train[field])
    median = np.median(train[field])
    train[field+'_mean_diff'] = train[field] - mean
    train[field+'_median_diff'] = train[field] - median
    #train[field+'_log'] = np.log(train[field]+1)
    test[field+'_mean_diff'] = test[field] - mean
    test[field+'_median_diff'] = test[field] - median
    #test[field+'_log'] = np.log(test[field]+1)
train.shape
# prepare data for modelling
from sklearn import metrics,svm
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.cross_validation import train_test_split, cross_val_score

X = train[list(set(train.columns)-set(['Unnamed: 0','y']) - set(cat_cols))]
y = train['y']

x_all = X
y_all = y

x_test = test[list(set(train.columns)-set(['Unnamed: 0','y']) - set(cat_cols))]

#splitting on test-train samples
x_train,x_val,y_train,y_val = train_test_split(X, y,test_size=0.3, random_state=0)

print (x_train.shape, x_val.shape, x_test.shape, x_all.shape)

x_train = x_train.fillna(-1)
x_test = x_test.fillna(-1)
x_val = x_val.fillna(-1)
# just try rf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

clf_rf = RandomForestClassifier(n_estimators = 700, random_state=0, max_depth = 10) 
clr_rf = clf_rf.fit(x_train, y_train)

print ("roc_auc_score train", roc_auc_score(y_train, clf_rf.predict_proba(x_train)[:,1]))
print ("roc_auc_score test", roc_auc_score(y_val, clf_rf.predict_proba(x_val)[:,1]))
# params for look
n_estimators = list(range(50, 800, 50))
max_depth = list(range(2, 10, 1))
max_features = ['sqrt', 'log2', 'auto']
from sklearn.grid_search import GridSearchCV

rf_cv = GridSearchCV(sklearn.ensemble.RandomForestClassifier(), {'n_estimators': n_estimators, 'max_depth': [2,3,4,5,6,7,8], 'max_features' : ['sqrt']}, cv = 5, n_jobs = -1, scoring='roc_auc')
rf_cv.fit(x_all, y_all)
rf_cv.best_params_
rf_cv.best_score_  
print ("roc_auc_score train", roc_auc_score(y_train, rf_cv.predict_proba(x_train)[:,1]))
print ("roc_auc_score test", roc_auc_score(y_val, rf_cv.predict_proba(x_val)[:,1]))

%%time
from sklearn.grid_search import GridSearchCV

gb_best = GridSearchCV(sklearn.ensemble.GradientBoostingClassifier(), {'n_estimators': [100,150,170,180,190,200,250], 'max_depth': [4,5,6,7,8,9,10], 'max_features' : ['sqrt']}, cv = 5, n_jobs = -1, scoring='roc_auc')
gb_best.fit(x_all, y_all)
gb_best.best_params_
gb_best.best_score_  
print ("roc_auc_score train", roc_auc_score(y_train, gb_best.predict_proba(x_train)[:,1]))
print ("roc_auc_score test", roc_auc_score(y_val, gb_best.predict_proba(x_val)[:,1]))


from sklearn.ensemble import AdaBoostClassifier

ada_best = GridSearchCV(sklearn.ensemble.AdaBoostClassifier(), {'n_estimators': [150,200,250,300,350,400], 'learning_rate': [0.01,0.1,0.2,0.3]}, cv = 5, n_jobs = -1, scoring='roc_auc')
ada_best.fit(x_all, y_all)
print (ada_best.best_params_)
print (ada_best.best_score_)
print ("roc_auc_score train", roc_auc_score(y_train, ada_best.predict_proba(x_train)[:,1]))
print ("roc_auc_score test", roc_auc_score(y_val, ada_best.predict_proba(x_val)[:,1]))

#data scaling for linear models
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)
x_train_norm = scaler.transform(x_train)
x_val_norm = scaler.transform(x_val)
x_test_norm = scaler.transform(x_test)

x_all_norm = scaler.transform(x_all)

data_train_norm = pd.DataFrame(data=x_train_norm, columns=x_train.columns)
data_val_norm = pd.DataFrame(data=x_val_norm, columns=x_train.columns)
data_test_norm = pd.DataFrame(data=x_test_norm, columns=x_train.columns)

data_all_norm = pd.DataFrame(data=x_all_norm, columns=x_train.columns)
print (data_train_norm.shape, data_val_norm.shape, data_test_norm.shape, data_all_norm.shape)
from sklearn.linear_model import LogisticRegression, SGDClassifier
#C=10, penalty='l1'
log_reg = LogisticRegression().fit(data_train_norm,y_train)

print ("roc_auc_score train", roc_auc_score(y_train, log_reg.predict_proba(data_train_norm)[:,1]))
print ("roc_auc_score test", roc_auc_score(y_val, log_reg.predict_proba(data_val_norm)[:,1]))

# data for ansambling
x_all_level2 = pd.DataFrame(index = x_all.index)
x_all_level2['rf_cv'] = rf_cv.predict_proba(x_all)[:,1]
x_all_level2['gb_best'] = gb_best.predict_proba(x_all)[:,1]
x_all_level2['ada_best'] = ada_best.predict_proba(x_all)[:,1]
x_all_level2['log_reg'] = log_reg.predict_proba(data_all_norm)[:,1]

x_test_level2 = pd.DataFrame(index = x_test.index)
x_test_level2['rf_cv'] = rf_cv.predict_proba(x_test)[:,1]
x_test_level2['gb_best'] = gb_best.predict_proba(x_test)[:,1]
x_test_level2['ada_best'] = ada_best.predict_proba(x_test)[:,1]
x_test_level2['log_reg'] = log_reg.predict_proba(data_test_norm)[:,1]
# ansamble

log_reg_ans_all = GridSearchCV(LogisticRegression(), {'C': [0.01,0.1,1,10], 'penalty': ['l1','l2']}, cv = 5, n_jobs = -1, scoring='roc_auc')
log_reg_ans_all.fit(x_all_level2, y_all)

print (log_reg_ans_all.best_params_)
print (log_reg_ans_all.best_score_)
print ("roc_auc_score all", roc_auc_score(y_all, log_reg_ans_all.predict_proba(x_all_level2)[:,1]))
# submission
my_subm = pd.DataFrame()
my_subm['id'] = test['Unnamed: 0']
my_subm['y'] = log_reg_ans_all.predict_proba(x_test_level2)[:,1]
my_subm.to_csv('my_subm_ans_new.csv', index = False)
my_subm.head()









