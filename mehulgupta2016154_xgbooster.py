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
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier as lgb
z=x['Survived']
x['Fare']=pd.cut(x['Fare'],bins=[-100,1,10,30,50,100,200,1000],labels=['invalid','least','less','average','abovAvg','High','veryHigh'])
x['Age']=pd.cut(x['Age'],bins=[-100,1,10,20,40,60,100],labels=['invalid','small','teen','adult','matured','old'])
y['Fare']=pd.cut(y['Fare'],bins=[-100,1,10,30,50,100,200,1000],labels=['invalid','least','less','average','abovAvg','High','veryHigh'])
y['Age']=pd.cut(y['Age'],bins=[-100,1,10,20,40,60,100],labels=['invalid','small','teen','adult','matured','old'])
x=x.drop(['PassengerId','Survived','Name','Ticket'],axis=1)
from sklearn.model_selection import train_test_split as tts
param={'n_estimators':[90,100,110],'learning_rate':[0.1,0.13,0.09],'max_depth':[5,6,7]}
knn={'n_neighbors':[3,4,5,6,7,8,9,10,11,12,13]}

p=lgb(max_depth=7)
from sklearn.preprocessing import LabelEncoder as le
for c in x.columns:
    if x[c].dtype=='object':
        x[c]=le().fit_transform(x[c].astype(str))
x.Age=le().fit_transform(x.Age.astype(str))
x.Fare=le().fit_transform(x.Fare.astype(str))
y.Age=le().fit_transform(y.Age.astype(str))
y.Fare=le().fit_transform(y.Fare.astype(str))

x=x.apply(lambda f:f.fillna(f.median()))
xtrain,xval,ztrain,zval=tts(x,z,train_size=0.7)
p.fit(xtrain,ztrain,eval_set=[(xtrain,ztrain),(xval,zval)],eval_metric='rmse',early_stopping_rounds=100)
for d in y.columns:
    if y[d].dtype=='object':
        y[d]=le().fit_transform(y[d].astype(str))

a=pd.DataFrame(p.predict(y[x.columns]))
a.index=y['PassengerId']
a.columns=['Survived']
a.index.name='PassengerId'
a
a.to_csv('result.csv')
