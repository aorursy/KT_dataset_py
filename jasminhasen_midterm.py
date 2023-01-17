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
import pickle

from sklearn.naive_bayes import GaussianNB

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
train_data = pd.read_csv('../input/santander-customer-transaction-prediction-dataset/train.csv')

test_data = pd.read_csv('../input/santander-customer-transaction-prediction-dataset/test.csv')
target = train_data['target'].astype(float)
train_data.drop(['ID_code', 'target'], axis = 1, inplace=True)

test_data.drop(['ID_code'], axis = 1, inplace=True )
data = {}

data['data'] = train_data

data['target'] = target

X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'])
gnb = GaussianNB()

gnb.fit(X_train, y_train)



with open('gaussianNB.pkl', 'wb') as file:  

    pickle.dump(gnb, file)

gnb = pickle.load(open('gaussianNB.pkl', 'rb'))



y_pred =  gnb.predict(X_test)



print('auc_roc score:', roc_auc_score(y_test, y_pred))
y_pred =  gnb.predict(test_data)
res = pd.DataFrame(pd.read_csv("../input/midterm-23860-kz/sample_submission.csv"))

res['target'] = pd.Series(y_pred,dtype=int)
y_pred = gnb.predict(test_data)



print('auc_roc score:', roc_auc_score(data['target'], y_pred))



print('probability of each class to happen:', gnb.class_prior_)

gnb
from sklearn.svm import SVC



svc = SVC(kernel='sigmoid', gamma=0.05)

svc.fit(X_train, y_train)



with open('SVC_sigmoid_gamma0.05.pkl', 'wb') as file:  

    pickle.dump(svc, file)
svc = pickle.load(open('SVC_sigmoid_gamma2.pkl', 'rb'))

y_pred = svc.predict(test_data)



res = pd.DataFrame(pd.read_csv('cmp_dataset/sample_submission.csv'))

res['target'] = pd.Series(y_pred,dtype=int)



res.to_csv('cmp_dataset/SVC_sigmoid_gamma2_result.csv', index = False, header=True)



print('auc_roc score:', roc_auc_score(target, y_pred))
svc.support_vectors_

svc.n_support_
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold

X_train = data['data']



xgb_result  = np.zeros(X_train.shape[0])

xgb_result  = np.zeros(X_train.shape[0])



pred = np.zeros(target.shape[0])



df_ids = np.array(X_train.index)    

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



skf.get_n_splits(data['data'].index, target)



model = xgb.XGBClassifier(max_depth=6,

                              learning_rate=1.5,

                              objective='binary:logistic', 

                              verbosity =.5,

                              eval_metric  = 'auc',

                              n_jobs=-2)

for counter, ids in enumerate(skf.split(df_ids, target)):

  X_fit, y_fit = X_train.values[ids[0]], target.values[ids[0]]

  X_val, y_val = X_train.values[ids[1]], target.values[ids[1]]



#   model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], eval_metric ='auc', early_stopping_rounds=5)#, verbose=False)

  

  model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], eval_metric='auc', early_stopping_rounds=12)#, verbose=False)



  xgb_result[ids[1]] += model.predict_proba(X_val)[:, 1]

  pred += model.predict_proba(test_data.values)[:,1]/skf.n_splits



with open('xgboost2_with_crutches.pkl', 'wb') as file:  

    pickle.dump(model, file)

print('auc_roc score:',roc_auc_score(target, pred))



res = pd.DataFrame(pd.read_csv('cmp_dataset/sample_submission.csv'))

res['target'] = pd.Series(pred,dtype=int)

res.to_csv('cmp_dataset/xgb_result.csv', index = False, header=True)
with open('xgboost1.pkl', 'wb') as file:  

    pickle.dump(model, file)

roc_auc_score(target, xgb_result)