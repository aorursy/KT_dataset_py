# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

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
y=x.drop(['Id'],axis=1)
a=y.count()>800
p=a[a==True]
z=list(p.index)

y=y[z]
y.shape
a=y.apply(pd.Series.nunique)<10
p=a[a==True]
p=list(p.index)
p
y=y.drop(['Street','LandContour','SaleCondition','SaleType','PoolArea','PavedDrive','GarageCond','GarageQual','Functional','KitchenAbvGr','BsmtHalfBath','Electrical','CentralAir','Heating','BsmtFinType2','BsmtCond','ExterCond','RoofMatl','RoofStyle','BldgType','Condition2','Condition1','LandSlope','Utilities'],axis=1)
y.columns
x=y.corr().iloc[:,-1]
h=x[(x>0.4) | (x<-0.4)] 
len(h)
from sklearn.preprocessing import OneHotEncoder
x=list(h.index)
y=y[x]
y=y.drop(['SalePrice'],axis=1)
z=pd.read_csv('../input/test.csv')
z=z[y.columns]
a=y.apply(lambda x:x.fillna(x.mean()))
b=z.apply(lambda x: x.fillna(x.mean()))
o=pd.read_csv('../input/train.csv')
import seaborn as sns
sns.distplot(o['SalePrice'])
sns.distplot(np.log(o['SalePrice']))
y
a
from sklearn.tree import DecisionTreeRegressor
l=DecisionTreeRegressor()
from sklearn.model_selection import cross_val_score
x=cross_val_score(l,a,np.log(o['SalePrice']),cv=5)
l.fit(a,np.log(o['SalePrice']))
x=l.predict(b)
n=[np.exp(v) for v in x]
x=pd.DataFrame(n)
x.index=pd.read_csv('../input/test.csv')['Id']
x.columns=['SalePrice']
x.to_csv('result.csv')
