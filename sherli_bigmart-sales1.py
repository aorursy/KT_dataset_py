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
import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score,mean_squared_error

import seaborn as sns

from math import sqrt



train = pd.read_csv("../input/Train.csv")
train.head()
train.shape
train.info()
train.describe()
train.isnull().sum()
train['Item_Fat_Content'].unique()
train['Item_Fat_Content'].value_counts()
train['Outlet_Establishment_Year'].unique().max()

train['Outlet_Establishment_Year'].unique().min()
train['Outlet_Age'] = 2020 - train['Outlet_Establishment_Year']

print(train['Outlet_Age'])
train.head()
train['Outlet_Size'].unique()
train['Outlet_Size'].mode()
train['Outlet_Size'].isna().value_counts()
train['Outlet_Size'] = train['Outlet_Size'].fillna("Medium")
train['Outlet_Size'].isna().value_counts()
train['Item_Weight'].isnull().value_counts()
train['Item_Weight'].mean()

print(train['Item_Weight'].mean())
train['Item_Weight'] =  train['Item_Weight'].fillna(train['Item_Weight'].mean())

train['Item_Weight'].isnull().any()
train['Item_Visibility'].hist(bins =20)


sns.boxplot(x = train['Item_Visibility'])
Q1 =  train['Item_Visibility'].quantile(0.25)

Q3 =  train['Item_Visibility'].quantile(0.75)

print(Q1)
print(Q3)
IQR =  Q3 - Q1

print(IQR)
filt_train =  train.query('(@Q1 - 1.5*@IQR) <= Item_Visibility <=(@Q3 + 1.5*@IQR)')

filt_train
filt_train.shape
train=filt_train
train.shape
train.info()
train['Item_Visibility_bins']  =  pd.cut(train['Item_Visibility'],[0.000,0.005,0.13,0.2],labels = ['Low Viz','Viz','High Viz'])
train['Item_Visibility_bins'].isna().value_counts()
train['Item_Visibility_bins'] = train['Item_Visibility_bins'].fillna("Low Viz")

train['Item_Visibility_bins'].isnull().any()
train['Item_Fat_Content'].value_counts()
train['Item_Fat_Content'] =  train['Item_Fat_Content'].replace(['low fat','LF'],'Low Fat')



train['Item_Fat_Content'] =  train['Item_Fat_Content'].replace(['reg'],'Regular')
train['Item_Fat_Content'].value_counts()

le = LabelEncoder()

train['Item_Fat_Content'].value_counts()

train['Item_Visibility_bins'].value_counts()
train['Outlet_Location_Type'].value_counts()
train['Outlet_Location_Type'].isnull().value_counts()


train['Item_Fat_Content'] =  le.fit_transform(train['Item_Fat_Content'])

train['Item_Visibility_bins'] =  le.fit_transform(train['Item_Visibility_bins'])

train['Outlet_Location_Type'] =  le.fit_transform(train['Outlet_Location_Type'])

train.info()
dummy = pd.get_dummies(train['Outlet_Type'])



dummy.head()
train = pd.concat([train,dummy],axis = 1)

train.isnull().any()
## Now taking only relenatvariables for model creation 



## Dropping irrelevant columns  





train = train.drop(['Item_Identifier','Item_Type','Outlet_Identifier','Outlet_Establishment_Year','Outlet_Size','Outlet_Type'],axis =1)
train.columns
train_corr=train.corr()
## PLotting Heat map for correlation  

sns.heatmap(data = train_corr,square =  True,cmap = 'bwr' )
X = train.drop('Item_Outlet_Sales',axis =1)

Y = pd.DataFrame(train.Item_Outlet_Sales)
X.columns
Y.columns
X.info()
from sklearn.model_selection import train_test_split



Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size = 0.3)



Xtrain.info()
## Model Number 1 , linear regression 

lin = LinearRegression()



lin.fit(Xtrain,Ytrain)
predictions =  lin.predict(Xtest)
# Now getting RMSE for linear regression model 



sqrt(mean_squared_error(Ytest,predictions))

lin.score(Xtest,Ytest)
from sklearn.linear_model import Ridge 



ridgereg = Ridge(alpha = 0.001,normalize = True)

ridgereg.fit(Xtrain,Ytrain)

pred_rig = ridgereg.predict(Xtest)

sqrt(mean_squared_error(Ytest,pred_rig))

ridgereg.score(Xtest,Ytest)
### Lasso model  

from sklearn.linear_model import Lasso



lassoreg = Lasso(alpha = 0.001,normalize = True)

lassoreg.fit(Xtrain,Ytrain)

pred_lasso = lassoreg.predict(Xtest) 

sqrt(mean_squared_error(Ytest,pred_lasso))

lassoreg.score(Xtest,Ytest)
from sklearn.linear_model import ElasticNet





elasreg = ElasticNet(alpha = 0.001,normalize = True)

elasreg.fit(Xtrain,Ytrain)

pred_elasreg = elasreg.predict(Xtest) 

sqrt(mean_squared_error(Ytest,pred_elasreg))

lassoreg.score(Xtest,Ytest)