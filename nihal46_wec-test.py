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
train=pd.read_csv('/kaggle/input/webclubrecruitment2019/TRAIN_DATA.csv',index_col='Unnamed: 0')

test=pd.read_csv('/kaggle/input/webclubrecruitment2019/TEST_DATA.csv')

ID=test['Unnamed: 0']

y=train['Class']

train.drop(columns=['Class'],inplace=True)

test.drop(columns=['Unnamed: 0'],inplace=True)
train.head()
missing=train.isnull().sum()

print(missing[missing>0])
cat_cols=['V2','V3','V4','V5','V7','V8','V9','V16','V11']

num_cols=set(train.columns)-set(cat_cols)

num_cols=list(num_cols)

print('no. of categorical columns is {}'.format(len(cat_cols)))

print('no. of numerical columns is {}'.format(len(num_cols)))

un=train[cat_cols].nunique()

print(un)
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split



OHE=OneHotEncoder(handle_unknown='ignore',sparse=False)



cat_features_train=pd.DataFrame(OHE.fit_transform(train[cat_cols]))

cat_features_test=pd.DataFrame(OHE.transform(test[cat_cols]))

cat_features_train.index=train.index

cat_features_test.index=test.index

print(cat_features_train.shape)



num_train=train[num_cols]

num_test=test[num_cols]



X=pd.concat([num_train,cat_features_train],axis=1)

X_test=pd.concat([num_test,cat_features_test],axis=1)

print(X.shape)

from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score





xgbcl=XGBClassifier(random_state=2)

kfold=StratifiedKFold(n_splits=3,shuffle=True,random_state=12)

gsc=GridSearchCV(xgbcl,param_grid={'max_depth':[3,4,5,6,8],'n_estimators':[50,100,150,200,250,300],'learning_rate':[0.1,0.5,0.01,0.05]},scoring='roc_auc',cv=kfold,verbose=1,n_jobs=-1)



grid_result=gsc.fit(X,y)

print('best set of parameters is {}'.format(grid_result.best_params_))



pred=gsc.predict_proba(X_test)

print(pred.shape)

result=pd.DataFrame({'Id': ID , 'PredictedValue' : pred[:,1] })

result.to_csv('output.csv',index = False)