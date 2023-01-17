%%time

%matplotlib inline

import numpy as np

import os

import glob

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import random

import xgboost as xgb

from sklearn.metrics import matthews_corrcoef



from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from xgboost import XGBRegressor

import lightgbm as lgb

from lightgbm import LGBMRegressor

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.cross_validation import StratifiedKFold

from sklearn.metrics import matthews_corrcoef, roc_auc_score

from sklearn.grid_search import RandomizedSearchCV

from catboost import CatBoostClassifier,CatBoostRegressor



from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder,LabelEncoder



from sklearn.svm import SVC

from sklearn.svm import SVR

from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor

from sklearn.feature_selection import VarianceThreshold



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.model_selection import KFold

from sklearn.metrics import r2_score,mean_squared_error

from math import sqrt

from scipy import stats

from scipy.stats import norm, skew #for some statistics

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV



from sklearn.metrics import matthews_corrcoef

from sklearn.metrics import f1_score

from sklearn.metrics import fbeta_score
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
train=pd.read_csv('../input/train_LZdllcl.csv')

test=pd.read_csv('../input/test_2umaH9m.csv')

sub=pd.read_csv('../input/sample_submission_M0L0uXE.csv')
train.describe()
train['education'].replace(np.nan,"Bachelor's",inplace=True)

test['education'].replace(np.nan,"Bachelor's",inplace=True)



train['education'].replace("Master's & above",3,inplace=True)

test['education'].replace("Master's & above",3,inplace=True)

train['education'].replace("Bachelor's",2,inplace=True)

test['education'].replace("Bachelor's",2,inplace=True)

train['education'].replace("Below Secondary",1,inplace=True)

test['education'].replace("Below Secondary",1,inplace=True)
train['previous_year_rating'].replace(np.nan,3.,inplace=True)

test['previous_year_rating'].replace(np.nan,3.,inplace=True)
train['sum_metric'] = train['awards_won?']+train['KPIs_met >80%'] + train['previous_year_rating']

test['sum_metric'] = test['awards_won?']+test['KPIs_met >80%'] + test['previous_year_rating']



train['tot_score'] = train['avg_training_score'] * train['no_of_trainings']

test['tot_score'] = test['avg_training_score'] * test['no_of_trainings']
from sklearn import preprocessing

le = preprocessing.LabelEncoder()



train['department'] = le.fit_transform(train['department'])

test['department'] = le.transform(test['department'])

train['region'] = le.fit_transform(train['region'])

test['region'] = le.transform(test['region'])

train['education'] = le.fit_transform(train['education'])

test['education'] = le.transform(test['education'])

train['gender'] = le.fit_transform(train['gender'])

test['gender'] = le.transform(test['gender'])



train['recruitment_channel'] = le.fit_transform(train['recruitment_channel'])

test['recruitment_channel'] = le.transform(test['recruitment_channel'])
Y1=train['is_promoted']

train1=train.drop(['employee_id','is_promoted','recruitment_channel'],axis=1)

train1=train1.values

Y=Y1.values



test_id=test['employee_id']

test1 = test.drop(['employee_id','recruitment_channel'],axis=1)

test1=test1.values
scaler = StandardScaler()

scaler.fit(train1)

train2 = scaler.transform(train1)

test2 = scaler.transform(test1)
pca = PCA(n_components=1)

pca.fit(train2)

train_pca = pca.transform(train2)

test_pca = pca.transform(test2)

train3=np.column_stack((train2,train_pca))

test3=np.column_stack((test2,test_pca))
#create the cross validation fold for different boosting and linear model.

from sklearn.cross_validation import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

SEED=42

# clf = lgb.LGBMClassifier()

st_train = train3

st_test = test3

# clf = xgb.XGBClassifier()

# Y=Y1

# clf = SVC(probability=True)

# clf = RandomForestClassifier(max_depth=4, random_state=0)

clf = lgb.LGBMClassifier(max_depth= 8, learning_rate=0.0941, n_estimators=197, num_leaves= 17, reg_alpha=3.4492 , reg_lambda= 0.0422) #lgb_pca

#clf = lgb.LGBMClassifier(max_depth= 8, learning_rate=0.0941, n_estimators=197, num_leaves= 17, reg_alpha=3.4492 , reg_lambda= 0.0422) #lgb_pca

# clf=CatBoostClassifier()

# clf = XGBClassifier()

# clf = Ridge()



# clf=ExtraTreesClassifier(n_estimators=10000, criterion='entropy', max_depth=9,  min_samples_leaf=1,  n_jobs=30, random_state=1)

# clf = xgb.XGBClassifier(random_state=42,colsample_bytree = 0.9279,gamma = 0.6494,learning_rate = 0.1573,max_depth = 7,min_child_weight = 6,n_estimators = 70,subsample = 0.6404)

# clf = RGFClassifier(max_leaf=500,algorithm="RGF",test_interval=100, loss="LS")

# clf = LogisticRegression()

# clf = LogisticRegression(class_weight ={1:4})



clf1 = lgb.LGBMClassifier(max_depth= 8, learning_rate=0.0941, n_estimators=197, num_leaves= 17, reg_alpha=3.4492 , reg_lambda= 0.0422) #lgb_pca

# clf2 = RGFClassifier(max_leaf=500,algorithm="RGF",test_interval=100, loss="LS")

clf3 = xgb.XGBClassifier(random_state=42,colsample_bytree = 0.9279,gamma = 0.6494,learning_rate = 0.1573,max_depth = 7,min_child_weight = 6,n_estimators = 70,subsample = 0.6404)



# clf = VotingClassifier(estimators=[('LR',clf2),('LGB',clf1),('LGB1',clf3)],voting='soft',

#                         weights=[3,4,2])



fold = 8

cv = StratifiedKFold(Y, n_folds=fold,shuffle=True, random_state=30)

X_preds = np.zeros(st_train.shape[0])

preds = np.zeros(st_test.shape[0])

for i, (tr, ts) in enumerate(cv):

    print(ts.shape)

    mod = clf.fit(st_train[tr], Y[tr])

    X_preds[ts] = mod.predict_proba(st_train[ts])[:,1]

    preds += mod.predict_proba(st_test)[:,1]

    print("fold {}, ROC AUC: {:.3f}".format(i, roc_auc_score(Y[ts], X_preds[ts])))

    predictions = [round(value) for value in X_preds[ts]]

    print(f1_score(Y[ts], predictions))

score = roc_auc_score(Y, X_preds)

print(score)

preds1 = preds/fold

# pick the best threshold out-of-fold

thresholds = np.linspace(0.01, 0.99, 50)

mcc = np.array([f1_score(Y, X_preds>thr) for thr in thresholds])

plt.plot(thresholds, mcc)

best_threshold = thresholds[mcc.argmax()]

print(mcc.max())

print(best_threshold)
##create the submission file.

prediction_rfc=list(range(len(preds1)))

for i in range(len(preds1)):

    prediction_rfc[i]=1 if preds1[i]>best_threshold else 0



sub = pd.DataFrame({'employee_id': test_id, 'is_promoted': prediction_rfc})

sub=sub.reindex(columns=["employee_id","is_promoted"])

sub.to_csv('submission.csv', index=False)
#lightgbm bayesian optimization

from sklearn.cross_validation import cross_val_score

from bayes_opt import BayesianOptimization



def xgboostcv(max_depth,learning_rate,n_estimators,num_leaves,reg_alpha,reg_lambda):

    cv = StratifiedKFold(Y, n_folds=8,shuffle=True, random_state=30)

    return cross_val_score(lgb.LGBMClassifier(max_depth=int(max_depth),learning_rate=learning_rate,n_estimators=int(n_estimators),

                                             silent=True,nthread=-1,num_leaves=int(num_leaves),reg_alpha=reg_alpha,

                                           reg_lambda=reg_lambda),

                           train3,Y,"roc_auc",cv=cv).mean()



xgboostBO = BayesianOptimization(xgboostcv,{'max_depth': (4, 10),'learning_rate': (0.001, 0.1),'n_estimators': (10, 1000),

                                  'num_leaves': (4,30),'reg_alpha': (1, 5),'reg_lambda': (0, 0.1)})



xgboostBO.maximize()

print('-'*53)

print('Final Results')

print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])
#xgboost bayesian optimization

from sklearn.cross_validation import cross_val_score

from bayes_opt import BayesianOptimization



def xgboostcv(max_depth,learning_rate,n_estimators,gamma,min_child_weight):

    cv = StratifiedKFold(Y, n_folds=8,shuffle=True, random_state=30)

    return cross_val_score(xgb.XGBClassifier(max_depth=int(max_depth),learning_rate=learning_rate,n_estimators=int(n_estimators),

                                             silent=True,nthread=-1,gamma=gamma,min_child_weight=min_child_weight),

                           train1,Y,"f1",cv=8).mean()



xgboostBO = BayesianOptimization(xgboostcv,{'max_depth': (4, 10),'learning_rate': (0.001, 0.3),'n_estimators': (50, 1000),

               'gamma': (0.01,1.0),'min_child_weight': (2, 10)})



xgboostBO.maximize()

print('-'*53)

print('Final Results')

print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])