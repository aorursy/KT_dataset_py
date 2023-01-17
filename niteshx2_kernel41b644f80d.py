# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df.describe()
df.info()
X = df.drop(['id' , 'target'] , axis = 1)

y = df['target']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
def modelfit(alg, dtrain, predictors, target , useTrainCV=True, cv_folds=5, early_stopping_rounds=5):

    if useTrainCV:

        xgb_param = alg.get_xgb_params()

        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,

            metrics='auc', early_stopping_rounds=early_stopping_rounds)

        alg.set_params(n_estimators=cvresult.shape[0])

    

    #Fit the algorithm on the data

    alg.fit(dtrain[predictors], dtrain[target],eval_metric='auc')

        

    #Predict training set:

    dtrain_predictions = alg.predict(dtrain[predictors])

    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

        

    #Print model report:

    print ("\nModel Report")

    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))

    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))

                    

    #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)

    #feat_imp.plot(kind='bar', title='Feature Importances')

    #plt.ylabel('Feature Importance Score')
from xgboost import XGBClassifier
from sklearn import metrics 

import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV

# Create the parameter grid: gbm_param_grid 

xgb4 = xgb.XGBClassifier(

 learning_rate =0.5,

 n_estimators=5,

 max_depth=4,

 min_child_weight=1,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 reg_alpha = 10.728910519108444,

 reg_lambda= 40.9847051755586085,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)





xgb1 = XGBClassifier(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=5,

 min_child_weight=1,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)



xgb3 = XGBClassifier(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=4,

 min_child_weight=6,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 reg_alpha=0.005,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)



xgb2 = XGBClassifier(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=4,

 min_child_weight=6,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

'''param_test1 = {

 'max_depth':range(3,10,2),

 'min_child_weight':range(1,6,2)

}

gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,

 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 

 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch1.fit(train[predictors],train[target])

gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_'''
from sklearn.model_selection import GridSearchCV

from xgboost.sklearn import XGBClassifier  

'''param_test1 = {

 'max_depth':range(3,10,2),

 'min_child_weight':range(1,6,2)

}

'''

'''gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,

 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 

 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch1.fit(X_train,y_train)

gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_'''
predictors = df.drop(['id', 'target'] , axis = 1)

predictors = predictors.columns
target = 'target'

modelfit(xgb4 , df[:180] , predictors , target)

modelfit(xgb1, df[:180] , predictors, target)

modelfit(xgb2, df[:180] , predictors, target)

modelfit(xgb3, df[:180] , predictors, target)



predictors
df_test.describe()
y_pred_xgb4 = xgb4.predict(df_test.drop(['id'] , axis = 1))

y_pred_xgb3 = xgb3.predict(df_test.drop(['id'] , axis = 1))

y_pred_xgb2 = xgb2.predict(df_test.drop(['id'] , axis = 1))

y_pred_xgb1 = xgb1.predict(df_test.drop(['id'] , axis = 1))
y_pred_xgb2
y_pred = pd.DataFrame(y_pred_xgb4*4 + y_pred_xgb3*2 + y_pred_xgb2*2 + y_pred_xgb1*2)/10

ans_XGB = pd.concat(( df_test['id'] , y_pred) , axis = 1)

ans_XGB.columns = ['id' , 'target']

outX = ans_XGB.to_csv('outX.csv' , index = False)





'''print ("Accuracy : %.4g" % metrics.accuracy_score(df[180:]['target'], y_pred_xgb))

print ("AUC Score (Train): %f" % metrics.roc_auc_score(df[180:]['target'], y_pred_xgb))'''
#pd.DataFrame(y_pred_xgb).describe()
'''ans_XGB = pd.concat(( df_test['ID_code'] , xgb) , axis = 1)

ans_XGB.columns = ['ID_code' , 'target']

outX = ansX.to_csv('outX.csv' , index = False)

'''
params = {#'num_leaves': 9,

         #'min_data_in_leaf': 42,

         'objective': 'binary',

         'max_depth': 5,

         #'learning_rate': 0.123,

         'boosting': 'gbdt',

         #'bagging_freq': 5,

         #'bagging_fraction': 0.8,

         #'feature_fraction': 0.8201,

         #'bagging_seed': 11,

         #'reg_alpha': 10.728910519108444,

         #'reg_lambda': 40.9847051755586085,

         'random_state': 42,

         'metric': 'auc',

         'verbosity': -1,

         'subsample': 0.81,

         #'min_gain_to_split': 0.01077313523861969,

         #min_child_weight': 19.428902804238373,

         'num_threads': 4}
from sklearn.model_selection import KFold

import time 

import lightgbm as lgb



folds = KFold(n_splits = 4 )

y_pred_lgb = np.zeros(len(X_test))

for fold_n, (train_index, valid_index) in enumerate(folds.split(X,y)):

    print('Fold', fold_n, 'started at', time.ctime())

    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    

    train_data = lgb.Dataset(X_train, label=y_train)

    valid_data = lgb.Dataset(X_valid, label=y_valid)

        

    lgb_model = lgb.train(params,train_data,num_boost_round=2000,#change 20 to 2000

                    valid_sets = [train_data, valid_data],verbose_eval=300,early_stopping_rounds = 200)##change 10 to 200

            

    y_pred_lgb += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)/5
y_pred_lgb = lgb_model.predict(df_test.drop(['id'] , axis = 1))

min(y_pred_lgb)
'''y_pred = pd.DataFrame(y_pred_xgb4*4 + y_pred_xgb3*2 + y_pred_xgb2*2 + y_pred_xgb1*2)/10

ans_XGB = pd.concat(( df_test['id'] , y_pred) , axis = 1)'''






df_LGB = pd.DataFrame(y_pred_lgb + y_pred_xgb4*4 + y_pred_xgb3*2 + y_pred_xgb2*2 + y_pred_xgb1*2)/11

ans_LGB = pd.concat((df_test['id'] , df_LGB) , axis = 1)

ans_LGB.columns = ['id' , 'target']

outX = ans_LGB.to_csv('outX.csv' , index = False)
