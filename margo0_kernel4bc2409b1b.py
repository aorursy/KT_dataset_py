#Importing necessary libraries and datasets

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.gridspec as gridspec

from datetime import datetime

from scipy.stats import skew  # for some statistics

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor

#from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

#from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from mlxtend.regressor import StackingCVRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt

import scipy.stats as stats

import sklearn.linear_model as linear_model

import matplotlib.style as style

import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#Importing training and test datasets

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
sample_submission.head()
#Sample train set

train.head()
#Sample test dataset

test.head()
print (f"Test has {test.shape[0]} rows and {test.shape[1]} columns")

print (f"Train has {train.shape[0]} rows and {train.shape[1]} columns")
train.get_dtype_counts()
train.isnull().sum().sort_values(ascending =False)
test.isnull().sum().sort_values(ascending =False)
sns.distplot(train['SalePrice']);

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')
fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
#skewness and kurtosis

print("Skewness: " + str(train['SalePrice'].skew()))

print("Kurtosis: " + str(train['SalePrice'].kurt()))
## Getting the correlation of all the features with target variable. 

(train.corr()**2)["SalePrice"].sort_values(ascending = False)[1:]
style.use('fivethirtyeight')

plt.subplots(figsize = (15,10))

## Plotting target variable with predictor variable(OverallQual)

sctplt = sns.scatterplot( train.OverallQual, train.SalePrice);

plt.show(sctplt)
style.use('fivethirtyeight')

plt.subplots(figsize = (15,10))

sctplt = sns.scatterplot( train.GrLivArea, train.SalePrice);

plt.show(sctplt)
plt.subplots(figsize = (15,10))

sctplt = sns.scatterplot( train.GarageArea, train.SalePrice);

plt.show(sctplt)
plt.subplots(figsize = (15,10))

sctplt = sns.scatterplot(train.TotalBsmtSF,train.SalePrice,);

plt.show(sctplt)
style.use('fivethirtyeight')

plt.subplots(figsize = (15,10))

sctplt = sns.scatterplot( train['1stFlrSF'], train.SalePrice);

plt.show(sctplt)
style.use('fivethirtyeight')

plt.subplots(figsize = (15,10))

sctplt = sns.scatterplot(  train.MasVnrArea, train.SalePrice);

plt.show(sctplt)
## Deleting those two values with outliers. 

train = train[train.GrLivArea < 4500]

train.reset_index(drop = True, inplace = True)



## save a copy of this dataset so that any changes later on can be compared side by side.

previous_train = train.copy()
## Plot sizing. 

fig, (ax1, ax2) = plt.subplots(figsize = (20,10), ncols=2,sharey=False)

## Scatter plotting for SalePrice and GrLivArea. 

sns.scatterplot( x = train.GrLivArea, y = train.SalePrice,  ax=ax1)

## Putting a regression line. 

sns.regplot(x=train.GrLivArea, y=train.SalePrice, ax=ax1)



## Scatter plotting for SalePrice and MasVnrArea. 

sns.scatterplot(x = train.MasVnrArea,y = train.SalePrice, ax=ax2)

## regression line for MasVnrArea and SalePrice. 

sns.regplot(x=train.MasVnrArea, y=train.SalePrice, ax=ax2);
plt.subplots(figsize = (15,10))

sns.residplot(train.GrLivArea, train.SalePrice);
## Customizing grid for two plots. 

fig, (ax1, ax2) = plt.subplots(figsize = (20,6), ncols=2, sharey = False, sharex=False)

## doing the first scatter plot. 

sns.residplot(x = previous_train.GrLivArea, y = previous_train.SalePrice, ax = ax1)

## doing the scatter plot for GrLivArea and SalePrice. 

sns.residplot(x = train.GrLivArea, y = train.SalePrice, ax = ax2);
## Plot fig sizing. 

style.use('ggplot')

sns.set_style('whitegrid')

plt.subplots(figsize = (30,20))

## Plotting heatmap. 



# Generate a mask for the upper triangle (taken from seaborn example gallery)

mask = np.zeros_like(train.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True





sns.heatmap(train.corr(), cmap=sns.diverging_palette(20, 220, n=200), mask = mask, annot=True, center = 0, );

## Give title. 

plt.title("Heatmap of all the Features", fontsize = 30);
fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
train.describe()