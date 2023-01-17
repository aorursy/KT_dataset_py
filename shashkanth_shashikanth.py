#shashikanth 

import os

print((os.listdir('../input/')))
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
df_train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')

df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')
test_index=df_test['Unnamed: 0'] #copying test index for later
tr = np.array(df_train.values[:,1:-1])



train_label = np.array(df_train.values[:,-1])

train_data = np.array(tr.astype(np.float))

test_data = np.array(df_test.values[:,1:])

trainX, trainY = train_data,train_label

testX = test_data

import xgboost as xgb

from imblearn.over_sampling import SMOTE

from xgboost.sklearn import XGBClassifier

smt = SMOTE()

X_train,y_train = smt.fit_sample(trainX,trainY)
def modelfit(alg, dtrain, y,useTrainCV=False, cv_folds=5, early_stopping_round=50):

    

    if useTrainCV:

        xgb_param = alg.get_xgb_params()

        xgtrain = xgb.DMatrix(dtrain, label=y)

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,

            metrics='auc', early_stopping_rounds=early_stopping_round, verbose_eval=True)

        alg.set_params(n_estimators=cvresult.shape[0])

        alg.fit(dtrain, y,eval_metric='auc')

        

    dtrain_predictions = alg.predict(dtrain)

    dtrain_predprob = alg.predict_proba(dtrain)

        

    y = np.eye(2)[y]





    return dtrain_predprob  


from xgboost import XGBClassifier



xgb1 = XGBClassifier(objective='binary:logistic')

modelfit(xgb1, X_train, y_train,True)
predic=modelfit(xgb1, testX,None)
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(predic[:,1])

result.head()
result.to_csv('output.csv', index=False)