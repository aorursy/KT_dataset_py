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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt 
%matplotlib inline
import os 
print(os.listdir("../input"))
Training=pd.read_csv("../input/train.csv")
TrainingSales=Training.corr()["SalePrice"].sort_values(ascending=False)
TrainingSales
TrainingNum=Training[["SalePrice","OverallQual","GrLivArea","GarageCars","TotalBsmtSF"]]
CorelationN=sns.pairplot(Training,height=5,
                    x_vars=["OverallQual","GrLivArea","GarageArea","TotalBsmtSF"],
                    y_vars=["SalePrice"],kind="reg")



CorelationN=sns.pairplot(Training,height=5,
                    x_vars=["GrLivArea","TotalBsmtSF"],
                    y_vars=["SalePrice"],kind="reg",hue="OverallQual")
x=Training["MSZoning"]
y=Training["SalePrice"]
sns.boxplot(x,y)
Training['MSZoningNum']=Training['MSZoning'].map({'FV':5,'RL':4,'RH':3,'RM':2,'C':1})
x=Training['MSZoningNum']
y=Training['SalePrice']
x.corr(y)
x=Training["BldgType"]
y=Training["SalePrice"]
sns.boxplot(x,y)


Training['BldgTypeNum']=Training['BldgType'].map({'TwnhsE':5,'1Fam':4,'Twnhs':3,'Duplex':2,'2fmCon':1})
x=Training['BldgTypeNum']
y=Training['SalePrice']
x.corr(y)
x=Training['HouseStyle']
y=Training['SalePrice']
sns.boxplot(x,y,width=0.7,
                 palette="colorblind",showmeans=True)


Training.HouseStyle.unique()
Training['HouseStyleNum']=Training['HouseStyle'].map({'2.5Fin':8,'2Story':7,'1Story':6,'SLv1':5,'2.5Unf':4,
                                                     '1.5Fin':3,'SFoyer':2,'1.5Unf':1})
x=Training['HouseStyleNum']
y=Training['SalePrice']
x.corr(y)
x=Training['Exterior1st']
y=Training['SalePrice']
sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(13, 6.5)

sns.boxplot(x,y)
Training.Exterior1st.unique()
Training['Exterior1stNum']=Training['Exterior1st'].map({'Stone':15,'lmStucc':15,'CemntBd':14,'VinylSd':13,'Plywood':12,
                                                        'BrkFace':11,'HdBoard':10,'Stucco':9,'Wd Sdng':7,
                                                        'MetalSd':6,'AsbShng':5,'BrkComm':4,'CBlock':3,'AsphShn':2})
x=Training['Exterior1stNum']
y=Training['SalePrice']
x.corr(y)

x=Training['BsmtFinType1']
y=Training['SalePrice']
sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(13, 6.5)

sns.boxplot(x,y)
Training['BsmtFinType1Num']=Training['BsmtFinType1'].map({'GLQ':6,'Unf':5,'LwQ':4,'ALQ':3,'Rec':1,'BLQ':1})
x=Training['BsmtFinType1Num']
y=Training['SalePrice']
x.corr(y)
x=Training['KitchenQual']
y=Training['SalePrice']
sns.set_style('ticks')
sns.boxplot(x,y)
Training['KitchenQualNum']=Training['KitchenQual'].map({'Ex':4,'Gd':3,'TA':2,'Fa':1})
x=Training['KitchenQualNum']
y=Training['SalePrice']
x.corr(y)
x=Training['PoolQC']
y=Training['SalePrice']


sns.boxplot(x,y)
Training['PoolQCNum']=Training['PoolQC'].map({'Ex':3,'Fa':2,'Gd':1})
x=Training['PoolQCNum']
y=Training['SalePrice']
x.corr(y)
x=Training['Neighborhood']
y=Training['SalePrice']
sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(15, 6.5)
sns.boxplot(x,y,showmeans=True,palette="Blues_d")
Training.Neighborhood.unique()
Training[['Neighborhood','SalePrice']].groupby('Neighborhood').median().sort_values(by='SalePrice',ascending=False)
Training['NeighborhoodNum']=Training['Neighborhood'].map({'NridgHt':25,'NoRidge':24,'StoneBr':23,'Timber':22,'Somerst':21,
                                             'Veenker':20,'Crawfor':19,'ClearCr':18,'CollgCr':17,'Blmngtn':16,
                                             'NMAmes':15,'Gilbert':14,'SawyerW':13,'Mitchel':12,'NPkVill':11,
                                             'NAmes':10,'SWISU':9,'Blueste':8,'Sawyer':7,'BrkSide':6,'Edwards':5,
                                             'OldTown':4,'Brdale':3,'IDOTRR':2,'MeadowV':1})
x=Training['NeighborhoodNum']
y=Training['SalePrice']
x.corr(y)


x=Training['Condition2']
y=Training['SalePrice']
sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(15, 6.5)
sns.boxplot(x,y,showmeans=True)
Training['Condition2Num']=Training['Condition2'].map({'PosA':8,'PosN':7,'PRAe':6,'Norm':5,'Feedr':4,'PRAn':3,
                                                     'Artery':2,'PRNn':1})
x=Training['Condition2Num']
y=Training['SalePrice']
x.corr(y)
HighCor=Training[['GrLivArea','NeighborhoodNum','KitchenQualNum','GarageArea','TotalBsmtSF','SalePrice']]
HighCor
