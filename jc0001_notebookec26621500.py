import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import sklearn

import random

 

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

from sklearn import cross_validation

from sklearn.cross_validation import train_test_split



from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

from sklearn.metrics import mean_squared_error, make_scorer

from scipy.stats import skew

from IPython.display import display



# Definitions

pd.set_option('display.float_format', lambda x: '%.3f' % x)





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



#-----------------------------------------------------------------------------

#                     Create Training & Testintg Set

#-----------------------------------------------------------------------------

csv =  pd.read_csv('../input/nyc-rolling-sales.csv')

train, test = sklearn.cross_validation.train_test_split(csv, train_size = 0.8)





#-----------------------------------------------------------------------------

#                     Data Analysis - Clean up

#-----------------------------------------------------------------------------

col = csv.columns

train['SALE PRICE'].describe()

sns.distplot(train['SALE PRICE']);

train = train.drop(train[train['SALE PRICE'] > 110000000].index) #Drop extreme values

train = train.drop(train[train['TOTAL UNITS'] < 1].index) #Drop null values



#correlation matrix (Check variables that are effecting Sales Price)

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);



#saleprice correlation matrix (Sorted by significance of correlation)

k = 12 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SALE PRICE')['SALE PRICE'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=2)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.1f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()  



#Scatter plots between 'SalePrice' and correlated variables 

sns.set()

cols = ['SALE PRICE', 'GROSS SQUARE FEET', 'TOTAL UNITS', 'RESIDENTIAL UNITS', 'COMMERCIAL UNITS']

sns.pairplot(train[cols], size = 2.5)

plt.show()



#standardizing data (Check significance of outliers: expect normal data to be around 0)

saleprice_scaled = StandardScaler().fit_transform(train['SALE PRICE'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)



#       Bivariate analysis SALE PRICE VS GROSS SQUARE FEET

#       check which points are off the trend (aka outliers)

var = 'GROSS SQUARE FEET'

data = pd.concat([train['SALE PRICE'], train[var]], axis=1)

data.plot.scatter(x=var, y='SALE PRICE');



#Identify the position of outliers then remove them

train.sort_values(by = 'GROSS SQUARE FEET', ascending = False)[:3]

train = train.drop(train[train['GROSS SQUARE FEET'] == 1021752].index)



#       Bivariate analysis SALE PRICE VS TOTAL UNITS

#       check which points are off the trend (aka outliers)

var = 'TOTAL UNITS'

data = pd.concat([train['SALE PRICE'], train[var]], axis=1)

data.plot.scatter(x=var, y='SALE PRICE');



train.sort_values(by = 'TOTAL UNITS', ascending = False)[:5]

train = train.drop(train[train['GROSS SQUARE FEET'] == 555954].index)



#-----------------------------------------------------------------------------

#                     Data Analysis - Transformation

#-----------------------------------------------------------------------------



#Normality Check on Sales price-heavy tail, not even close to normal

sns.distplot(train['SALE PRICE'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['SALE PRICE'], plot=plt)



#apply lognormal transformation (Still not normal, but alot better)

train['SALE PRICE'] = np.log(train['SALE PRICE'])



#Apply lognormal transformation on other factors for the same reason

train['GROSS SQUARE FEET'] = np.log(train['GROSS SQUARE FEET'])

sns.distplot(train['GROSS SQUARE FEET'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['SALE PRICE'], plot=plt)



train['TOTAL UNITS'] = np.log(train['TOTAL UNITS'])

sns.distplot(train['TOTAL UNITS'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['SALE PRICE'], plot=plt)



#-----------------------------------------------------------------------------

#                     Data Analysis - Regression

#-----------------------------------------------------------------------------

y = train['SALE PRICE']

grossSize = train['GROSS SQUARE FEET'] 

unitNum = train['TOTAL UNITS']

borough = train['BOROUGH']#Caterlogical Vars





# Linear Regression

lr = LinearRegression()

lr.fit(grossSize, y)
