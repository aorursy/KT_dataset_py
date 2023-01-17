# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
pd.pandas.set_option('display.max_columns',None)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df.head()
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(df,df['SalePrice'],test_size=0.1,random_state=0)
print(xtrain.shape)
print(xtest.shape)
catfeanan=[fea for fea in df.columns if df[fea].isnull().sum()>1 and df[fea].dtypes=='O']
print(catfeanan)
def replace_cat_fea(df,catfeanan):
    dat=df.copy()
    dat[catfeanan]=dat[catfeanan].fillna('Missing')
    return dat

df=replace_cat_fea(df,catfeanan)
df[catfeanan].isnull().sum()
df.head()
numfeanan=[fea for fea in df.columns if df[fea].isnull().sum()>1 and df[fea].dtypes!='O']
print(numfeanan)
for fea in numfeanan:
    med_val=df[fea].median()
    
    df[fea+'nan']=np.where(df[fea].isnull(),1,0)
    df[fea].fillna(med_val,inplace=True)
    
    
df[numfeanan].isnull().sum()
df.head()
for fea in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
    df[fea]=df['YrSold']-df[fea]
df[['YearBuilt','YearRemodAdd','GarageYrBlt']].head()
num_fea=['LotFrontage','LotArea','1stFlrSF','GrLivArea','SalePrice']
for fea in num_fea:
    df[fea]=np.log(df[fea])
    plt.hist(df[fea])
    plt.show()
catfea=[fea for fea in df.columns if df[fea].dtypes=='O']
catfea
for fea in catfea:
    temp=df.groupby(fea)['SalePrice'].count()/len(df)
    dftemp=temp[temp>0.01].index
    df[fea]=np.where(df[fea].isin(dftemp),df[fea],'RareValue')
df.head(50)
for feature in catfea:
    labels_ordered=df.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    df[feature]=df[feature].map(labels_ordered)
df.head()
from sklearn.preprocessing import MinMaxScaler
feascale=[fea for fea in df.columns if fea not in ['Id','SalePrice']]
scale=MinMaxScaler()
scale.fit(df[feascale])
dat=pd.concat([df[['Id','SalePrice']].reset_index(drop=True),pd.DataFrame(scale.transform(df[feascale]),columns=feascale)],axis=1)
dat.head()
dat.to_csv('traindat.csv',index=False)
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
df=pd.read_csv('traindat.csv')
df.head()
ytrain=df[['SalePrice']]
xtrain=df.drop(['Id','SalePrice'],axis=1)
fea_model=SelectFromModel(Lasso(alpha=0.005,random_state=0))
fea_model.fit(xtrain,ytrain)
fea_model.get_support()
sel_fea=xtrain.columns[(fea_model.get_support())]
print('Total No of Feature',xtrain.shape[1])
print('NO of selected feature',len(sel_fea))
print(sel_fea)
xtrain=xtrain[sel_fea]
xtrain.head()
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(xtrain,ytrain,test_size=0.2)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
mod=reg.fit(xtrain,ytrain)
mod.score(xtest,ytest)
