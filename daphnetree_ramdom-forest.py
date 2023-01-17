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
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV

import lightgbm as lgb

import xgboost as xgb
train = pd.read_csv("../input/santander-value-prediction-challenge/train.csv")

test = pd.read_csv("../input/santander-value-prediction-challenge/test.csv")
#top 208 feature importance using xgbregressor

cols1=['ID','f190486d6', '15ace8c9f', '861076e21', '66ace2992', '58e2e02e6', 'eeb9cd3aa', '414b74eaa', 'c47340d97', '024c577b9',

 '64dd02e44', 'ff3ebf76b', 'b8a716ebf', 'f514fdb2e', '578eda8e0', '1931ccfdd', '1702b5bf0', 'a396ceeb9',

 '491b9ee45', '9fd594eec', '2288333b4', '20aa07010', '0c8063d63', '324921c7b', 'ba136ae3f', '61c1b7eb6',

 '58e056e12', '59cafde1f', '5324862e4', '26fc93eb7', 'd83a2b684', '6cf7866c1', 'aecaa2bc9', 'dd84674d0',

 '9f494676e', 'fb387ea33', '2c339d4f2', '7121c40ee', '899dbe405', '06a1c3b47', '20604ed8f', 'bb1113dbb',

 '22fbf6997', 'cc62f0df8', '29ab304b9', 'd24a55c98', 'f58fb412c', '342e7eb03', 'e222309b0', '703885424',

 '26628e8d8', 'c380056bb', '402b0d650', '241f0f867', '2ad744c57', '62e59a501', '6eef030c1', '8337d1adc',

 '371da7669', '6f88afe65', '1f8a823f2', 'ed8ff54b5', '58ed8fb53', '193a81dce', '24018f832', 'd3b9b9a70',

 '44f3640e4', '58232a6fb', 'afe8cb696', '3e1100230', 'c86c0565e', 'b22288a77', 'caa9883f6', '64e38e7a2',

 '9306da53f', 'fb0f5dbfe', '5c6487af1', '8618bc1fd', '18cad608c', 'cb4f34014', 'edc84139a', 'b0c0f5dae',

 '6619d81fc', 'a39758dae', '8d8276242', 'bc70cbc26', '854e37761', '4edc3388d', '8b710e161', '29181e29a',

 '2ec5b290f', 'f8cd9ae02', '9f7b782ac', 'b9ba17eb6', '26ab20ff9', 'f8405f8b9', '8675bec0b', '6df033973',

 '1d9078f84', '20551fa5b', '22d7ad48d', '1184df5c2', 'bee629024', '41bc25fef', 'c3f400e36', '0c9462c08',

 'e0bb9cf0b', '45f6d00da', '45226872a', '17b81a716', 'f9847e9fe', '0e3ef9e8f', 'c5a231d81', 'b30e932ba',

 'a3da2277a', 'b0e45a9f7', 'd37030d36', 'bdf773176', 'ced6a7e91', '7ec8cff44', '9c502dcd9', '5661462ee',

 '2c42b0dce', '935ca66a9', '2322dbbbb', 'ca96df1db', 'f74e8f13d', '009319104', '609784003', 'c944a48b5',

 'aa164b93b', 'e613715cc', '4fe8b17c2', '55f4891bb', 'ce6349807', '3c4df440f', '30b3daec2', 'd6bb78916',

 '4da206d28', '2e55d0383', 'cbeddb751', '16b532cdc', '08e89cc54', 'c270cb02b', '64e483341', 'b6fa5a5fd',

 '6c3d38537', 'db147ffca', 'ff2c9aa8f', '5239ceb39', 'ba852cc7a', '191e21b5f', '190db8488', '86558e595',

 '2c6c62b54', '4416cd92c', '07c9d1f37', 'f6eba969e', 'b791ce9aa', 'e2c21c4bc', 'df838756c', '8479174c2',

 'c3c633f64', '84d9d1228', '77697c671', 'd5754aa08', 'b96718230', 'be4729cb7', '3af1785ee', 'b43a7cfd5',

 '24b2da056', 'd428161d9', '2cb73ede7', '05c9b6799', '26144d11f', '38e6f8d32', '05f54f417', '2a83c3267',

 'b4cfe861f', '823ac378c', '5fb9cabb1', 'e8a3423d6', '087e01c14', '1c62e29a7', '69e1143e2', '092271eb3',

 'fb5a3097e', 'b98f3e0d7', '5f341a818', '5030aed26', '56896bb36', 'ad009c8b9', '87a2d8324', 'a3ef69ad5',

 '1d04efde3', '467bee277', 'fc436be29', '55dbd6bcb', '686d60d8a', 'e13b0c0aa', 'b6daeae32', '82f715995',

 '84ec1e3db', '0a03426de', 'd66bbb5ed', 'e4159c59e', '68153d35e', 'de0aaf6f4', '177993dc6']
test=test[cols1]

cols1.insert(1,'target')

train=train[cols1]
all_zero_columns=[i for i in train.columns if train[i].nunique()==1]

train=train[[i for i in train.columns if i not in all_zero_columns]]

test=test[[i for i in test.columns if i not in all_zero_columns]]

duplicte_columns=['d60ddde1b', 'acc5b709d', '912836770', 'f8d75792f', 'f333a5f60']

train=train[[i for i in train.columns if i not in duplicte_columns]]

test=test[[i for i in test.columns if i not in duplicte_columns]]

X = np.log1p(train.drop(["ID", "target"], axis=1))

y = np.log1p(train["target"].values)

test = np.log1p(test.drop(["ID"], axis=1))
rf1=RandomForestRegressor(oob_score=True)

#para1={'n_estimators': [50],'max_depth': [2]}

para1={'n_estimators': [50,100,200,300,400,500],'max_depth': [8,10,12,16,20]}

gs1 = GridSearchCV(rf1,para1,cv = 5,scoring = 'neg_mean_squared_error',verbose=5)

gs1.fit(X,y)

print(gs1.best_params_)
rf2=RandomForestRegressor(max_depth=gs1.best_params_['max_depth'],n_estimators=gs1.best_params_['n_estimators'],oob_score=True)

para2={'max_features':['auto', 'sqrt']}

gs2 = GridSearchCV(rf2,para2,cv = 5,scoring = 'neg_mean_squared_error',verbose=5)

gs2.fit(X,y)

print(gs2.best_params_)
rf3=RandomForestRegressor(max_depth=gs1.best_params_['max_depth'],n_estimators=gs1.best_params_['n_estimators'],max_features=gs2.best_params_['max_features'],oob_score=True)

para3={'min_samples_split': [i/10.0 for i in range(1,8)],'min_samples_leaf': [i/10.0 for i in range(1,6)]}

#para1={'n_estimators': [50,100,200,300,400,500],'max_depth': [8,10,12,16,20]}

gs3 = GridSearchCV(rf3,para3,cv = 5,scoring = 'neg_mean_squared_error',verbose=5)

gs3.fit(X,y)

print(gs3.best_params_)
rfbest=RandomForestRegressor(max_depth=gs1.best_params_['max_depth'],n_estimators=gs1.best_params_['n_estimators'],

                             max_features=gs2.best_params_['max_features'],

                             min_samples_split=gs3.best_params_['min_samples_split'],min_samples_leaf=gs3.best_params_['min_samples_leaf'],

                             oob_score=True)
rfbest
rfbest.fit(X, y)

y_pred = rfbest.predict(test)

sub = pd.read_csv('../input/santander-value-prediction-challenge/sample_submission.csv')

sub["target"] = np.expm1(y_pred)

sub.to_csv('submit_rf.csv', index=False)