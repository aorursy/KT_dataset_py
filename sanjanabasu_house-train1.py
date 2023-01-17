import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')
df_train=pd.read_csv('../input/house-prices-dataset/train.csv')
df_train.head()
sns.boxplot(df_train['SalePrice'])
sns.distplot(df_train['SalePrice'])
sns.distplot(df_train['LotArea'])
sns.distplot(df_train['SalePrice'],bins=100,kde=True)
cols=['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF']
plt.figure(figsize=(20,10))
corrmat=df_train.corr()
sns.heatmap(corrmat)
cols=['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF']
sns.pairplot(df_train[cols])
sns.regplot(data=df_train,y='SalePrice',x='YearBuilt') #regression analysis
sns.lmplot(data=df_train,y='SalePrice',x='YearBuilt')
sns.stripplot(data=df_train,y='SalePrice',x='SaleCondition')
sns.swarmplot(data=df_train,y='SalePrice',x='SaleCondition')
sns.jointplot(df_train['LotFrontage'],df_train['LotArea'])