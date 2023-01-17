# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  #graphcal representation
import seaborn as sns  # data visualization 
from sklearn.preprocessing import StandardScaler  
import warnings
#warnings.filterwarnings('ignore')
#%matplotlib inline


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Load the data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.columns)  # show the columns for dataset
print(test.count())   #Count the uniqe entries
print( train['SalePrice'].describe())  # describe the mentioned column
print(train.shape)
plt.plot(train['SalePrice'])
plt.show()
train['SalePrice'].plot.hist(alpha=0.5)
plt.show()
sns.distplot(train['SalePrice'])
print('Skewness: %f'  % train['SalePrice'].skew() )
print('Kurtosis: %f' % train['SalePrice'].kurt())
def relationship(var):
    temp = pd.concat([train['SalePrice'], train[var]], axis=1)
    #plt.title(var)
    print(temp.head())
    temp.plot.scatter(x=var, y='SalePrice')  #scatter plot
    #plt.show()
    #sns.boxplot(x=var, y=train['SalePrice'], data=temp)  #box plot seaborn
plt.show()
relationship('GrLivArea')
#relationship('TotalBsmtSF')
#relationship('OverallQual')
#relationship('YearBuilt')

#Correlation matrix
cor_train = train.corr()  #Compute pairwise correlation of columns, excluding NA/null values
print(cor_train.shape)
print(train.shape)
print(train.count())
print(cor_train.count())
plt.subplots(figsize=(12, 9))  #zooming the plot
sns.heatmap(cor_train, vmax=1, square=True)  #heat map
#'SalePrice' correlation matrix (zoomed heatmap style)
k=10
cols_train = cor_train.nlargest(k, 'SalePrice')['SalePrice'].index  #assiing 10 largest matching clomn name 
cm = np.corrcoef(train[cols_train].values.T)
cols_train
print(cm.shape)
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols_train.values, xticklabels=cols_train.values)
#Scatter plots between 'SalePrice' and correlated variables
sns.set()
corr_variables = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[corr_variables], size=4)  #plosting the multiple plots same as plt subplot
plt.show()
#print(train.isnull().sum().nlargest(20))    #all the largest missing count 
total_missing = train.isnull().sum().sort_values(ascending=False)          #missing count in sorted manner
percent_missing = (train.isnull().sum()/train.isnull().count())
train_data_missing = pd.concat([total_missing,percent_missing], axis=1, keys = ['Total','% missing']).sort_values('Total', ascending=False)
train_data_missing.head(20)
#dropping colums with null value
wm_train = train.drop((train_data_missing[train_data_missing['Total']>1]).index,1)
wm_train = wm_train.drop(wm_train[wm_train['Electrical'].isnull()].index)
wm_train.isnull().sum().max()
#standardizing data
scaled_Saleprice = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis])  
print(scaled_Saleprice.shape)
print('outer range (low) of the distribution:')
print(scaled_Saleprice[scaled_Saleprice[:,0].argsort()][:10])
print('\nouter range (high) of the distribution:')
print(scaled_Saleprice[scaled_Saleprice[:,0].argsort()[-10:]])
#bivariate analysis saleprice/grlivarea
relationship('GrLivArea')
#deleting points
train_sort_GrLivArea = train.sort_values(by = 'GrLivArea', ascending = False)
print(train_sort_GrLivArea.head())
train = train.drop(train[train['Id'] == 1299 ].index)
train = train.drop(train[train['Id'] == 524 ].index)
train.shape
relationship('GrLivArea')
from scipy.stats import norm  #for normalisation
sns.distplot(train['SalePrice'], fit=norm)
plt.figure()
from scipy import stats  #for probability distribution
stats.probplot(train['SalePrice'], plot=plt)
train['SalePrice'] = np.log(train['SalePrice'])  #cahnging to log
sns.distplot(train['SalePrice'], fit=norm)
plt.figure()
stats.probplot(train['SalePrice'], plot=plt)
sns.distplot(train['GrLivArea'], fit=norm)
plt.figure()
stats.probplot(train['GrLivArea'], plot=plt)
train['GrLivArea'] = np.log(train['GrLivArea'])
sns.distplot(train['GrLivArea'], fit=norm)
plt.figure()
stats.probplot(train['GrLivArea'], plot=plt)
#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
train['HasBsmt']= pd.Series(len(train['TotalBsmtSF']), index=train.index)
train['HasBsmt'] = 0
train.loc[train['TotalBsmtSF']>0,'HasBsmt'] = 1
train.loc[train['HasBsmt']==1,'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])
sns.distplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
plt.figure()
stats.probplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
plt.scatter(train['SalePrice'], train['GrLivArea'])
train_dummy = pd.get_dummies(train)
train_dummy.head()
