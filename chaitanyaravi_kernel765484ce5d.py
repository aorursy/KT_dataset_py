# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

Train_DataFrame = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
Train_DataFrame.head(10)
Train_DataFrame.shape
Train_DataFrame.size
Train_DataFrame.columns
missing_count=Train_DataFrame.isna().sum()
missing_count[missing_count>0]
Train_DataFrame.info()
Train_DataFrame['SalePrice'].skew()
Train_DataFrame['SalePrice'].kurt()
Train_DataFrame['SalePrice'].describe()
Train_DataFrame.select_dtypes(include='float64')
Train_DataFrame.drop(['FireplaceQu','PoolQC','Fence','MiscFeature','BsmtUnfSF'], axis = 'columns', inplace = True)
df_num = Train_DataFrame.select_dtypes(include = ['float64', 'int64'])

df_num.head()

df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
x=Train_DataFrame['GrLivArea']

y=Train_DataFrame['SalePrice']

plt.xlabel('GrLivArea')

plt.ylabel('SalePrice')

plt.title('my graph')



plt.scatter(x,y,color='green',linewidth=2,linestyle='dashed')
#box plot overallqual/saleprice

var = 'OverallQual'

data = pd.concat([Train_DataFrame['SalePrice'], Train_DataFrame[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
df_num_corr = df_num.corr()['SalePrice'][:-1] # -1 because the latest row is SalePrice

golden_features_list = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False)

print("There is {} strongly correlated values with SalePrice:\n{}".format(len(golden_features_list), golden_features_list))
for i in range(0, len(df_num.columns), 5):

    sns.pairplot(data=df_num,

                x_vars=df_num.columns[i:i+5],

                y_vars=['SalePrice'])
#correlation matrix

corrmat = Train_DataFrame.corr()

f, ax = plt.subplots(figsize=(12, 15))

sns.heatmap(corrmat, vmax=.8, square=True,annot_kws={'size': 10});
numeric_features = Train_DataFrame.select_dtypes(include=[np.number])

correlation = numeric_features.corr()

print(correlation['SalePrice'].sort_values(ascending = False),'\n')

k= 11

cols = correlation.nlargest(k,'SalePrice')['SalePrice'].index

print(cols)

cm = np.corrcoef(Train_DataFrame[cols].values.T)

f , ax = plt.subplots(figsize = (14,12))

sns.heatmap(cm, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',

            linecolor="white",xticklabels = cols.values ,annot_kws = {'size':12},yticklabels = cols.values)
import seaborn as sns

import scipy.stats as st

from sklearn import ensemble, tree, linear_model

import missingno as msno

msno.matrix(Train_DataFrame)
#histogram and normal probability plot

from scipy import stats

from scipy.stats import norm

sns.distplot(Train_DataFrame['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(Train_DataFrame['SalePrice'], plot=plt)