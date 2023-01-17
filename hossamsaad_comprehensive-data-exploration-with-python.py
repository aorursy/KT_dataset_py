#invite people for the Kaggle party

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
#bring in the six packs

train = pd.read_csv('../input/train.csv')

train.head()

train.info()
train.isna().sum().sort_values(ascending = False)[0:20 ,]
#check the decoration

train.columns
#descriptive statistics summary

train['SalePrice'].describe()


#histogram

sns.distplot(train['SalePrice']) ;
#skewness and kurtosis

print("Skewness: %f" % train['SalePrice'].skew())  # %F (of float) is to replace a float Skew.

print("Kurtosis: %f" % train['SalePrice'].kurt())  #
#scatter plot grlivarea/saleprice

var = 'GrLivArea'

Area_saleprice = pd.concat([train['SalePrice'], train[var]] ,axis = 1)

Area_saleprice.plot(x= var , y = 'SalePrice' , ylim = (0.800000) ,kind = 'Scatter')

plt.show()

print(Area_saleprice.head().sort_values(var , ascending = True))
#scatter plot totalbsmtsf/saleprice

var = 'TotalBsmtSF'

bsmtsf_saleprice = pd.concat([train['SalePrice'], train[var]], axis=1)

bsmtsf_saleprice.plot(x=var, y='SalePrice', ylim=(0,800000),kind= 'Scatter')

plt.show()

print(bsmtsf_saleprice.head(10))
#box plot overallqual/saleprice

var = 'OverallQual'

Quality_saleprice= pd.concat([train['SalePrice'] , train[var]],axis = 1)

axiss = plt.subplots(figsize = (8,6))

fig=sns.boxplot(x= var , y = 'SalePrice' ,data= Quality_saleprice)

fig.axis(ymin = 0 , ymax = 800000);

var = 'YearBuilt'

yearB_saleprice = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=yearB_saleprice)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=45);
#correlation matrix

corrmatrix = train.corr()

f ,ax =plt.subplots(figsize=(15,11))

sns.heatmap(corrmatrix , vmax= .8 , square = True);
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmatrix.nlargest(k, 'SalePrice')['SalePrice'].index  #largest correlations only 

cm = np.corrcoef(train[cols].values.T) #correlation matrix transpose

f ,ax =plt.subplots(figsize=(15,11))

sns.set(font_scale=1.25) # font size

hm = sns.heatmap(cm, cbar=True, #Color legand bar 

                 annot=True, #write values of each cells.

                 square=True,#To resahpe into Squares 

                 fmt='.2f', #String Foramt of each string cells.

                 annot_kws={'size': 10}, #cell values size.

                 yticklabels=cols.values, 

                 xticklabels=cols.values)

plt.show()

#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], size = 2.5)

plt.show();
#missing data

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)



#dealing with missing data

train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)

train = train.drop(train.loc[train['Electrical'].isnull()].index)

train.isnull().sum().max() #just checking that there's no missing data missing...

#standardizing data

scalling_saleprice = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis]); #column vectors

lowRange = scalling_saleprice[scalling_saleprice[:,0].argsort()][:10]

highRange= scalling_saleprice[scalling_saleprice[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(lowRange)

print('\nouter range (high) of the distribution:')

print(highRange)



#bivariate analysis saleprice/grlivarea

var = 'GrLivArea'

Area_saleprice = pd.concat([train['SalePrice'], train[var]], axis=1)

Area_saleprice.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#deleting points

print(train.sort_values(by = 'GrLivArea', ascending = False)[:2]) #get last two big values

train = train.drop(train[train['Id'] == 1299].index)

train = train.drop(train[train['Id'] == 524].index)


print(train.sort_values(by = 'GrLivArea', ascending = False)[:2])

train = train.drop(train[train['Id'] == 1299].index)
#After detection outlier

var = 'GrLivArea'

Area_saleprice = pd.concat([train['SalePrice'], train[var]], axis=1)

Area_saleprice.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#bivariate analysis saleprice/grlivarea

var = 'TotalBsmtSF'

bsmtsf_saleprice = pd.concat([train['SalePrice'], train[var]], axis=1)

bsmtsf_saleprice.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#histogram and normal probability plot

sns.distplot(train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)
#applying log transformation

train['SalePrice'] = np.log(train['SalePrice'])
#transformed histogram and normal probability plot

sns.distplot(train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)
#histogram and normal probability plot

sns.distplot(train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['GrLivArea'], plot=plt)
#data transformation

train['GrLivArea'] = np.log(train['GrLivArea'])
#transformed histogram and normal probability plot

sns.distplot(train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['GrLivArea'], plot=plt)
#histogram and normal probability plot

sns.distplot(train['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['TotalBsmtSF'], plot=plt)
#create column for new variable (one is enough because it's a binary categorical feature)

#if area>0 it gets 1, for area==0 it gets 0

train['HasBsmtsf'] = pd.Series(len(train['TotalBsmtSF']), index=train.index)

train['HasBsmtsf'] = 0 

train.loc[train['TotalBsmtSF']>0,'HasBsmtsf'] = 1
#transform data

train.loc[train['HasBsmtsf']==1,'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])

#histogram and normal probability plot

sns.distplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)

#scatter plot

plt.scatter(train['GrLivArea'], train['SalePrice']);  #مدي التجانس بينهم بغد اللوغارتيم
#scatter plot

plt.scatter(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], train[train['TotalBsmtSF']>0]['SalePrice']);
#convert categorical variable into dummy

train = pd.get_dummies(train)

train.head(10)
