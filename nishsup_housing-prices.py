# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
# numerical feautres
X1=train.select_dtypes(exclude='object')
x1=test.select_dtypes(exclude='object')
for c in X1.columns:
    X1[c].fillna(X1[c].mean(),inplace=True)
for c in x1.columns:
    x1[c].fillna(x1[c].mean(),inplace=True)
#non numerical featurs 
X2=train.select_dtypes(include='object')
x2=test.select_dtypes(include='object')
for c in X2.columns:
    X2[c].fillna('NA',inplace=True)
    x2[c].fillna('NA',inplace=True)
X1.drop('SalePrice',axis=1,inplace=True)
X1.drop('Id',axis=1,inplace=True)
x1.drop('Id',axis=1,inplace=True)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X1=pd.DataFrame(sc.fit_transform(X1))
x1=pd.DataFrame(sc.transform(x1))
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
T=X2.append(x2)
for c in X2.columns:
    le.fit(T.loc[:,c])
    X2.loc[:,c]=le.transform(X2.loc[:,c])
    x2.loc[:,c]=le.transform(x2.loc[:,c])
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
T=X2.append(x2)
ohe.fit(T)
X3=ohe.transform(X2)
x3=ohe.transform(x2)
X3=pd.DataFrame(X3)
x3=pd.DataFrame(x3)
X=X1.join(X3,lsuffix='L_',rsuffix='R_')
x=x1.join(x3, lsuffix='L_',rsuffix='R_')
Y=train.SalePrice
#!nvidia-smi
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
Xeval, Xcv, Yeval, Ycv = train_test_split(X,Y,test_size = 0.2,random_state=0)
#tree_method='gpu_hist
from sklearn.metrics import mean_squared_error
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import numpy as np
def objective(space):
    # Instantiate the classifier
    clf = xgb.XGBRegressor(n_estimators =1000,colsample_bytree=space['colsample_bytree'],
                           learning_rate = .3,
                            max_depth = int(space['max_depth']),
                            min_child_weight = space['min_child_weight'],
                            subsample = space['subsample'],
                           gamma = space['gamma'],
                           reg_lambda = space['reg_lambda'])
    
    eval_set  = [( Xeval, Yeval), ( Xcv, Ycv)]
    
    # Fit the classsifier
    clf.fit(Xeval, Yeval,
            eval_set=eval_set, eval_metric="rmse",
            early_stopping_rounds=10,verbose=False)
    
    # Predict on Cross Validation data
    pred = clf.predict(Xcv)
    
    # Calculate our Metric - accuracy
    accuracy = mean_squared_error(Ycv, pred)
    #print(accuracy)

    # return needs to be in this below format. We use negative of accuracy since we want to maximize it.
    return {'loss': accuracy, 'status': STATUS_OK }
space ={'max_depth': hp.quniform("x_max_depth", 4, 40, 1),
        'min_child_weight': hp.quniform ('x_min_child', 1, 30, 1),
        'subsample': hp.uniform ('x_subsample', 0.5, 1),
        'gamma' : hp.uniform ('x_gamma', 0.1,0.7),
        'colsample_bytree' : hp.uniform ('x_colsample_bytree', 0.5,1),
        'reg_lambda' : hp.uniform ('x_reg_lambda', 0,1)
    }
trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)
print(best)
rf=xgb.XGBRegressor(x_colsample_bytree= 0.8950116320780152, x_gamma= 0.16299249472231195, x_max_depth= 37.0, x_min_child=5.0, x_reg_lambda= 0.6984614939647829, x_subsample= 0.886264032265592)
rf.fit(X,Y)
rf.score(X,Y)
y=rf.predict(x)
y=rf.predict(x)
ub=pd.DataFrame({'Id':test.Id,'SalePrice':y})
ub.to_csv('rf.csv',index=False)
