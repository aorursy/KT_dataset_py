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
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_train.columns
df_train
#descriptive statistics summary
df_train['SalePrice'].describe()
#histogram
sns.distplot(df_train['SalePrice'])
#skewness and kurtosis
print("Skewness of Saleprice: %f" % df_train['SalePrice'].skew())
print("Kurtosis of Saleprice: %f" % df_train['SalePrice'].kurt())
#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,900000))
#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,900000))
#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(10,6))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=900000)
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16,8))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ylim=0, ymax=900000)
plt.xticks(rotation=90)
#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(10,6))
sns.heatmap(corrmat, vmax=.8, square=True)




