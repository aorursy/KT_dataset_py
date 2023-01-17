# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/train.csv")
df.head()
#first check the number of column and rows in test data
df.shape
df.info()
#Describe will provide the statistical information about each column like min,max,std
df.describe()
#select the numerical for EDA
num_col=df.select_dtypes(exclude=['object'])
num_col.shape
#To know the corelation between variables
import matplotlib.pyplot as plt
import seaborn as sns
fig,ax=plt.subplots(figsize=(12,7))
corr_mat=num_col.corr()
sns.heatmap(corr_mat, fmt="g", cmap='viridis')
plt.show()
#most important step in model building is data preparation.
null_per=(df.isnull().sum()/len(df))
null_per=null_per[null_per>0]
null_data=pd.DataFrame(null_per)
null_data
# in above ouput Alley,Fence,MiscFeature,PoolQC are having above 90% of data is null
columns=['Fence','MiscFeature','Alley','PoolQC']
#df.head()
df=df.drop(columns,inplace=False,axis=1)
df.head()
df1=df.dropna(subset=['MasVnrType','MasVnrArea','Electrical','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond'])
df1.shape
df1=df1.fillna(method='ffill')
df1=df1.dropna(axis=0)
#df1.dtypes
#p=df1.isnull().sum()
plt.scatter(x=df1.LotArea,y=df1.SalePrice)
plt.xlabel('Sale Price')
plt.ylabel('LotArea')
plt.show()
