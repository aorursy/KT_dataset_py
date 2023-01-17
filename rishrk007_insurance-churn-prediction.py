# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.metrics import log_loss

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

import xgboost as xgb
import lightgbm as lgb

train=pd.read_csv("/kaggle/input/Train.csv")
test=pd.read_csv("/kaggle/input/Test.csv")
train.head()
train.isnull().sum()
train.describe()
train["labels"].value_counts().plot.bar()
del train["feature_12"]
del train["feature_10"]

del train["feature_5"]
del train["feature_6"]
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
corr = train.corr()
sns.heatmap(corr,annot=True)
features = list(set(train.columns)-set(['labels']))
target = 'labels'
len(features)
from sklearn.metrics import f1_score 
def metric(y,y0):
    return f1_score(y,y0)

def cross_valid(model,train,features,target,cv=3):
    results = cross_val_predict(model, train[features], train[target], method="predict",cv=cv)
    return metric(train[target],results)


models = [lgb.LGBMClassifier(), xgb.XGBClassifier(), GradientBoostingClassifier()
             ]

for i in models:
    model = i
    error = cross_valid(model,train,features,target,cv=10)
    print(str(model).split("(")[0], error)
train.head()
y=train["labels"]
del train["labels"]
x=train
def xgb_model(x,y, plot=True):
    evals_result = {}
    trainX, validX, trainY, validY = train_test_split(x,y, test_size=0.2, random_state=13)
    print("XGB Model")
    
    dtrain = xgb.DMatrix(trainX, label=trainY)
    dvalid = xgb.DMatrix(validX, label=validY)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    
    MAX_ROUNDS=2000
    early_stopping_rounds=100
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'learning_rate': 0.01,
        'num_round': MAX_ROUNDS,
        'max_depth': 8,
        'seed': 25,
        'nthread': -1,
        'num_class':5
    }
    
    model = xgb.train(
        params,
        dtrain,
        evals=watchlist,
        num_boost_round=MAX_ROUNDS,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=50
        #feval=metric_xgb
    
    )
    
    print("Best Iteration :: {} \n".format(model.best_iteration))
    
    
    if plot:
        # Plotting Importances
        fig, ax = plt.subplots(figsize=(24, 24))
        xgb.plot_importance(model, height=0.4, ax=ax)

xgb_model(x,y,plot=True)
xgb1 = xgb.XGBClassifier(
    booster='gbtree',
    objective='multi:softprob',
    learning_rate= 0.1,
    num_round= 1149,
    max_depth=8,
    seed=25,
    nthread=-1,
    eval_metric='mlogloss',
    num_class=5

)
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

print("X_train dataset: ", X_train.shape)
print("y_train dataset: ", y_train.shape)
print("X_test dataset: ", X_test.shape)
print("y_test dataset: ", y_test.shape)
print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))

xgb1.fit(X_train_res,y_train_res)
pred=xgb1.predict(X_test)
print(f1_score(pred,y_test))
def learning_rate_010_decay_power_099(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_010_decay_power_0995(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.995, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_005_decay_power_099(current_iter):
    base_learning_rate = 0.05
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3

import lightgbm as lgb
fit_params={"early_stopping_rounds":30, 
            "eval_metric" : 'auc', 
            "eval_set" : [(X_test,y_test)],
            'eval_names': ['valid'],
            #'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
            'verbose': 100,
            'categorical_feature': 'auto'}

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
param_test ={'num_leaves': sp_randint(6, 50), 
             'min_child_samples': sp_randint(100, 500), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}
n_HP_points_to_test = 100

import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

#n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum
clf = lgb.LGBMClassifier(max_depth=-1, random_state=314, silent=True, metric='None', n_jobs=4, n_estimators=5000)
gs = RandomizedSearchCV(
    estimator=clf, 
    param_distributions=param_test, 
    n_iter=n_HP_points_to_test,
    scoring='accuracy',
    cv=3,
    refit=True,
    random_state=314,
    verbose=True)
gs.fit(X_train_res, y_train_res, **fit_params)
print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))
opt_parameters =  {'colsample_bytree': 0.404828563763895, 'min_child_samples': 350, 'min_child_weight': 0.001, 'num_leaves': 36, 'reg_alpha': 1, 'reg_lambda': 5, 'subsample': 0.9274016358891894} 

clf_sw = lgb.LGBMClassifier(**clf.get_params())
#set optimal parameters
clf_sw.set_params(**opt_parameters)
gs_sample_weight = GridSearchCV(estimator=clf_sw, 
                                param_grid={'scale_pos_weight':[1,2,6,12]},
                                scoring='roc_auc',
                                cv=5,
                                refit=True,
                                verbose=True)


gs_sample_weight.fit(X_train_res, y_train_res, **fit_params)
print('Best score reached: {} with params: {} '.format(gs_sample_weight.best_score_, 
                                                       gs_sample_weight.best_params_))

clf_final = lgb.LGBMClassifier(**clf.get_params())
#set optimal parameters
clf_final.set_params(**opt_parameters)
#Train the final model with learning rate decay
clf_final.fit(X_train_res, y_train_res, **fit_params, callbacks=[lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_0995)])


pred=clf_final.predict(X_test)
print(f1_score(pred,y_test))
del test["feature_5"]
del test["feature_6"]
del test["feature_12"]
del test["feature_10"]

predf=xgb1.predict(test)
sub=pd.read_excel("/kaggle/input/sample_submission.xlsx")
sub["labels"]=predf
sub.to_excel("s1.xlsx",index=False)
from IPython.display import FileLink
FileLink(r's1.xlsx')
