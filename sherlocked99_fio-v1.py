# # This Python 3 environment comes with many helpful analytics libraries installed

# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# # For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import metrics   #Additional scklearn functions

from sklearn.model_selection import train_test_split, GridSearchCV

import xgboost as xgb

from xgboost import plot_importance

from xgboost.sklearn import XGBClassifier

import os

import matplotlib.pyplot as plt



target = 'targets'

IDcol = 'unique_id'
df = pd.read_csv("../input/train.csv")
df['targets'] = df['targets'] - 1

df.shape
# print(df['targets'])

df.drop(columns = ['x_82','x_51','x_61','x_6'], inplace = True) # removing bottom four features from version 1 

df
df.shape
def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    

    if useTrainCV:

        xgb_param = alg.get_xgb_params()

        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,

            metrics='mlogloss', early_stopping_rounds=early_stopping_rounds)

        alg.set_params(n_estimators=cvresult.shape[0])

    

    #Fit the algorithm on the data

    alg.fit(dtrain[predictors], dtrain['targets'],eval_metric='mlogloss')

        

    #Predict training set:

    dtrain_predictions = alg.predict(dtrain[predictors])

    dtrain_predprob = alg.predict_proba(dtrain[predictors])

        

    #print (dtrain_predprob.shape)

    #Print model report:

    print ("\nModel Report")

    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['targets'].values, dtrain_predictions))

    print ("Mlogloss Score (Train): %f" % metrics.log_loss(dtrain['targets'], dtrain_predprob))

    

    fig,ax = plt.subplots(figsize = (20,20))

    plot_importance(alg,ax=ax)

    #plt.show()

    return alg
predictors = [x for x in df.columns if x not in [target, IDcol]]

xgb1 = XGBClassifier(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=5,

 min_child_weight=1,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'multi:softprob',

 num_class = 9,

 nthread=4,

 scale_pos_weight=1,

 seed=27)



model = modelfit(xgb1, df, predictors)
d_test = pd.read_csv("../input/test.csv")

d_test.drop(columns = ['x_82','x_51','x_61','x_6'], inplace = True)

predictors = [x for x in d_test.columns if x not in [IDcol]]

dtest_predictions = model.predict(d_test[predictors])

dtest_predprob = model.predict_proba(d_test[predictors])



dtest_predprob.shape
submission = pd.read_csv("../input/sample_submission.csv")
i = 0

for col in submission.columns:

    if col=='unique_id':

        continue

    else:

        submission[col] = dtest_predprob[:,i]

    i = i+1
submission.to_csv("mysubmission2.csv",index = False)