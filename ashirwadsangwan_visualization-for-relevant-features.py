import matplotlib.pyplot as plt
import scipy.stats as stats
import lightgbm as lgb
import seaborn as sns
import xgboost as xgb
import pandas as pd
import numpy as np
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test.shape
train.shape
#train.info()
#test.info()
train.head()
train.describe()
train.isnull().sum() # It returns the number of null values in each column.
train.columns 
num_feat=train.select_dtypes(include=[np.number])
cat_feat=train.select_dtypes(include=[np.object])
#num_feat
#cat_feat
sns.distplot(train['SalePrice']);
train['SalePrice'].describe()
#Skewness and Kurtosis for Target Variable
print('Skewness :',train['SalePrice'].skew())
print('Kurtosis :',train['SalePrice'].kurt())
train.corr()['SalePrice']
rel_feat = ['SalePrice','OverallQual','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF',
                           'GrLivArea','FullBath','TotRmsAbvGrd','GarageCars','GarageArea']
rel_feat_corr = train.corr()['SalePrice'][['OverallQual','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF',
                                      'GrLivArea','FullBath','TotRmsAbvGrd','GarageCars','GarageArea']]
plt.figure(figsize = (15,15))
sns.heatmap(train[rel_feat].corr(),annot = True,linewidths = 0.5);
plt.savefig('Correlation Heatmap.png')
rel_feat_corr
plt.figure(figsize=(15,15))



plt.subplot(331)
plt.scatter(train['YearBuilt'], train['SalePrice'], color = 'b');
plt.title('YearBuilt vs SalePrice', fontsize = 14);
plt.subplot(332)
plt.scatter(train['YearRemodAdd'], train['SalePrice'], color = 'r');
plt.title('YearRemoAdd vs SalePrice');
plt.subplot(333)
plt.scatter(train['TotalBsmtSF'], train['SalePrice'], color = 'y');
plt.title('TotalNsmtSF vs SalePrice');
plt.subplot(334)
plt.scatter(train['1stFlrSF'], train['SalePrice'], color = 'm');
plt.title('1stFlrSF vs SalePrice');
plt.subplot(335)
plt.scatter(train['GrLivArea'], train['SalePrice'], color = 'orange');
plt.title('GrLivArea vs SalePrice');
plt.subplot(336)
plt.scatter(train['GarageArea'], train['SalePrice'], color = 'c');
plt.title('GarageArea vs SalePrice');
plt.subplot(337)
plt.scatter(train['TotRmsAbvGrd'], train['SalePrice'], color = 'k');
plt.title('TotRmsABvGrd vs SalePrice');
plt.subplot(338)
plt.scatter(train['GarageCars'], train['SalePrice'], color = 'purple');
plt.title('GarageCars vs SalePrice');
plt.subplot(339)
plt.scatter(train['FullBath'], train['SalePrice'], color = 'g');
plt.title('FullBath vs SalePrice');


plt.savefig('HousingPrices.png')

plt.figure(figsize= (15,7))
sns.boxplot(train['OverallQual'], train['SalePrice']);
plt.savefig('OverallQual Vs SalePrice.png')
plt.figure(figsize = (10,7))
plt.scatter(train['LotFrontage'], train['SalePrice']);  #It has a correlation of 0.35 with saleprice.
plt.figure(figsize= (15,8))
sns.boxplot(train['TotRmsAbvGrd'], train['SalePrice']);
sns.stripplot(train["TotRmsAbvGrd"],train["SalePrice"], jitter=True, edgecolor="gray")
plt.savefig('TotRmsAbvGrd Vs SalePrice.png')

#Sample size is decreasing after Total rooms above grade reaches to 10.
plt.figure(figsize= (15,8))
sns.boxplot(train['GarageCars'], train['SalePrice']);
sns.stripplot(train["GarageCars"],train["SalePrice"], jitter=True, edgecolor="gray")
plt.savefig('GarageCars Vs SalePrice.png')
#Median Sale Price going down after 4 Garagecars is undestandable after plotting the points on boxes.
plt.figure(figsize= (15,8))
sns.boxplot(train['FullBath'], train['SalePrice']);
plt.savefig('FullBath Vs SalePrice.png')
del train['Id']  #Removing the Id column from the data
train[rel_feat].isna().any()  
train[rel_feat].isna().any()
