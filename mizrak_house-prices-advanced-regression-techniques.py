# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib.pyplot as plt 

import seaborn as sns

import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.metrics import  r2_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/train.csv")

df.fillna((df.mean()), inplace=True)



veri=pd.DataFrame()

veri=df.filter(['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold','SalePrice'],axis=1)



plt.figure(figsize=(20, 16))



k = 37    

corr = veri.corr()



cols = corr.nlargest(k,'SalePrice')['SalePrice'].index



cm = np.corrcoef(veri[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cmap="YlGnBu",cbar=True, annot=True, square=True, fmt='.2f',

                 annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()


x=pd.DataFrame()

x=df.filter(['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd','MasVnrArea','GarageYrBlt','Fireplaces','BsmtFinSF1','LotFrontage','WoodDeckSF','2ndFlrSF','OpenPorchSF'],axis=1)

x.set_index(df['Id'],inplace=True)



df['SalePrice'].astype("float")

y=df['SalePrice']

y.index=df['Id']





from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.25,random_state = 42)



reg = LinearRegression()

reg.fit(x_train, y_train)



y_predicted = reg.predict(x_test)

print('R²: %.2f' % r2_score(y_test, y_predicted))
fig, ax = plt.subplots(figsize=(10,10))

ax.scatter(y_test, y_predicted)

ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)

ax.set_xlabel('measured')

ax.set_ylabel('predicted')

plt.show()





df2=pd.read_csv("../input/test.csv")

df2.fillna((df.mean()), inplace=True)

df3=df2.filter(['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd','MasVnrArea','GarageYrBlt','Fireplaces','BsmtFinSF1','LotFrontage','WoodDeckSF','2ndFlrSF','OpenPorchSF'],axis=1)

test_tahmin = reg.predict(df3)

tahmin=pd.DataFrame(test_tahmin)

tahmin.set_index(df2['Id'],inplace=True)



print(tahmin)