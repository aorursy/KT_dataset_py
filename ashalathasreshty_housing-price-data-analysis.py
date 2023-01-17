# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats
dtrain=pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
dtrain.info()
dtrain.columns
null_columns = dtrain.columns[dtrain.isnull().any()]

dtrain[null_columns].isnull().sum()
null_columns
df_nullcols = pd.DataFrame(dtrain[null_columns])
df_nullcols
df_workingdata = dtrain.copy()
df_workingdata.drop(df_workingdata[null_columns], axis = 1, inplace = True) 
df_workingdata.info()
df_workingdata.dtypes
df_workingdata.head()
dftr_num = df_workingdata.select_dtypes(np.number)

#dftr_num = df_workingdata.select_dtypes('int64')
dftr_num.info()

dftr_num.head()
dftr_cat = df_workingdata.select_dtypes('object')
dftr_cat.info()

dftr_cat.head()
dftr_num

dftr_num.info()
# Removing the 'Id' column which is not useful for statistical analysis

dfnum = dftr_num.drop(['Id'], axis = 1)

dfnum


chart = sns.distplot(dfnum["SalePrice"], color="teal")

plt.setp(chart.get_xticklabels(), rotation=45)



sns.set(style="white", color_codes=True)

sns.jointplot(x=dfnum["LotArea"], y=dfnum["SalePrice"], kind='scatter', s=200, color='m', edgecolor="skyblue", linewidth=2)

 
sns.regplot(x=dfnum["SalePrice"], y=dfnum["LotArea"], fit_reg=False)

plt.show()

data = dfnum[['SalePrice', 'OverallQual', 'TotalBsmtSF', 'YearBuilt', 'YrSold']]
chart = sns.pairplot(data, kind="scatter", hue = "YrSold", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))

plt.show()

data = dfnum[['SalePrice', 'MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea']]



f, axes = plt.subplots(1, 2, figsize=(15, 6))

fig = sns.boxplot(x=data['OverallQual'], y = data['SalePrice'], data=data, ax=axes[0])

fig = sns.boxplot(x=data['OverallCond'], y = data['SalePrice'], data=data, ax=axes[1])

fig.axis(ymin=0, ymax=800000);

sns.swarmplot(x=data['MSSubClass'], y=data['SalePrice'], data=data, hue='MSSubClass')

plt.legend(bbox_to_anchor=(1, 1), loc=2)

size = 11

corrmat = data.iloc[:,:size].corr()

f, ax = plt.subplots(figsize = (10,8))

sns.heatmap(corrmat,vmax=0.8,square=True);
data = dfnum[['MSSubClass', 'YearBuilt', 'YearRemodAdd', 'YrSold', 'SalePrice']]



f, axes = plt.subplots(1, 2, figsize=(15, 10))

fig = sns.boxplot(x=data['MSSubClass'], y = data['SalePrice'], data=data, ax=axes[0])

fig = sns.boxplot(x=data['YrSold'], y = data['SalePrice'], data=data, ax=axes[1])

fig.axis(ymin=0, ymax=800000);

plt.show()

f, axes = plt.subplots(2, 1, figsize=(15, 12))

fig2 = sns.boxplot(x=data['YearBuilt'], y = data['SalePrice'], data=data, ax=axes[0])

fig2.axis(ymin=0, ymax=800000);

plt.setp(fig2.get_xticklabels(), rotation=45)

fig3 = sns.boxplot(x=data['YearRemodAdd'], y = data['SalePrice'], data=data, ax=axes[1])

fig3.axis(ymin=0, ymax=800000);

plt.setp(fig3.get_xticklabels(), rotation=45)

plt.show()
dfcat = dftr_cat.copy()
# adding SalePrice column from numerical data to this categorical data





dfcat['SalePrice'] = pd.Series(dfnum['SalePrice'])



dfcat['OverallQual'] = pd.Series(dfnum['OverallQual'])

dfcat
data = dfcat[['LandContour', 'Condition1', 'BldgType', 'SalePrice']]



f, axes = plt.subplots(1, 2, figsize=(15, 10))

sns.swarmplot(x=data['LandContour'], y=data['SalePrice'], data=data, hue='BldgType', ax=axes[0])

plt.legend(bbox_to_anchor=(1, 1), loc=2)



sns.swarmplot(x=data['Condition1'], y=data['SalePrice'], data=data, hue='BldgType', ax=axes[1])

plt.legend(bbox_to_anchor=(1, 1), loc=2)
