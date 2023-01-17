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
x=pd.read_csv('../input/train.csv')
y=pd.read_csv('../input/test.csv')
x.columns
z=x['SalePrice']
x=x.drop(['Id','SalePrice','PoolQC','MiscFeature'],axis=1)
y=y.drop(['Id'],axis=1)
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
param={'learning_rate':[0.1,0.15,0.18,0.08,0.05],'n_estimators':[90,95,100,105,110],'max_depth':[5,6,7,8,9,11]}
q=GridSearchCV(XGBRegressor(),param)
from sklearn.preprocessing import LabelEncoder as le
from sklearn.preprocessing import OneHotEncoder as ohe

for c in x.columns:
    if x[c].dtype=='object':
        e=le().fit_transform(x[c].astype(str))
        x[c]=ohe(sparse=False).fit_transform(e.reshape(len(e),1))
        e=le().fit_transform(y[c].astype(str))
        y[c]=ohe(sparse=False).fit_transform(e.reshape(len(e),1))
        x[c]=x[c].fillna(x[c].median())
        y[c]=y[c].fillna(y[c].median())
        
q.fit(x,np.log(z))
pd.DataFrame(np.exp(q.predict(y[x.columns])),index=pd.read_csv('../input/test.csv')['Id'],columns=['SalePrice']).to_csv('result.csv')
