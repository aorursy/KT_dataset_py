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



df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

#check the decoration

df_train.columns

#histogram

sns.distplot(df_train['SalePrice']);

#skewness and kurtosis

print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())

#correlation matrix

corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);

#correlation matrix

corrmatt = df_test.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmatt, vmax=.8, square=True);
#saleprice correlation matrix

k = 20 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#saleprice correlation matrix

k = 20 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()



#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', '1stFlrSF','FullBath', 'TotRmsAbvGrd','YearBuilt','YearRemodAdd','Fireplaces','WoodDeckSF','2ndFlrSF','OpenPorchSF','HalfBath']

sns.pairplot(df_train[cols], size = 2.5)

plt.show();
import statsmodels.api as sm

#reread the data

df_train = pd.read_csv('../input/train.csv')





cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', '1stFlrSF','FullBath', 'TotRmsAbvGrd','YearBuilt','YearRemodAdd','Fireplaces','WoodDeckSF','2ndFlrSF','OpenPorchSF','HalfBath']

#normalize the skewed population with logs

df_train['SalePrice'] = np.log(df_train['SalePrice'])

df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

df_train['TotalBsmtSF'] = np.log(df_train['TotalBsmtSF']+1)

df_train['1stFlrSF'] = np.log(df_train['1stFlrSF']+1)

df_train['WoodDeckSF'] = np.log(df_train['WoodDeckSF']+1)

df_train['2ndFlrSF'] = np.log(df_train['2ndFlrSF']+1)

df_train['OpenPorchSF'] = np.log(df_train['OpenPorchSF']+1)

model = sm.OLS(df_train['SalePrice'], df_train[cols])

results = model.fit()

print(results.summary())

#betas

results.params

reconstr=np.dot(df_train[cols],results.params)

np.exp(reconstr)

#normalise identically the test

#rerread the data

df_test = pd.read_csv('../input/test.csv')



df_test['GrLivArea'] = np.log(df_test['GrLivArea'])

df_test['TotalBsmtSF'] = np.log(df_test['TotalBsmtSF']+1)

df_test['1stFlrSF'] = np.log(df_test['1stFlrSF']+1)

df_test['WoodDeckSF'] = np.log(df_test['WoodDeckSF']+1)

df_test['2ndFlrSF'] = np.log(df_test['2ndFlrSF']+1)

df_test['OpenPorchSF'] = np.log(df_test['OpenPorchSF']+1)



Qtest = np.dot(df_test[cols],results.params)





for pp in range(0,15):

    print(df_test['Id'][pp],np.exp(Qtest[pp]))

    