# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

# evaluation: RMSE = log(predicted value) - log(observed value)
describe_df = train.describe()

train['YrSold'].describe()

len(train.columns)
sp = train['SalePrice'].copy()

print(sp.describe())

print(sp.median())

train.get_dtype_counts()

# histogram of SalePrice

sp_skew = train['SalePrice'].skew()

print(sp_skew)

sns.set(font_scale = 1)

plt.xticks([0,200000,400000,6000000,800000])

sns.distplot(train['SalePrice'])

plt.show()
stats.probplot(sp, plot = plt, rvalue = True)

plt.show()
# log transform due to right skew

sp_transform = np.log(train['SalePrice'].copy())

# print(sp_transform)

sns.distplot(sp_transform) # histogram of saleprice after transformation

plt.show()

stats.probplot(sp_transform,plot = plt, rvalue = True)

plt.show()
# define list of columns that are num vs categorical

num = list(train.loc[:,train.dtypes != 'object'].drop('Id',axis = 1).columns.values)

cat = list(train.loc[:,train.dtypes == 'object'].columns.values)
# correlation analysis - want to see what variables have highest correlation with SalePrice & each other

# finding cross-correlation can prevent over fitting

corr = train.corr()

fig, ax = plt.subplots(figsize=(15,15))





SP_corr = corr.loc[:,['SalePrice']]

top_30_corr = SP_corr.nlargest(17,'SalePrice')

highest_corr = top_30_corr.index[0:17]

highest_corr = list(highest_corr)

train_corr = train.loc[:,highest_corr]

sns.set(font_scale = 2)

ax.set_title("Highest Correlated Variables")

sns.heatmap(train_corr.corr(), linewidth = .1, annot = True, cmap = 'Purples', annot_kws={"size": 15})

plt.show()
# pairplots of OverallQual and GrLivArea

sns.set(font_scale = 1)

sns.regplot(x = 'OverallQual', y = 'SalePrice', data = train, scatter = True, fit_reg = False)

plt.show()

sns.regplot(x = 'GrLivArea', y = 'SalePrice', data = train, scatter = True, color = 'g', fit_reg = False)

plt.show()
missing = train.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True, ascending = False)

missing
new = train['PoolQC']

new_2 = pd.get_dummies(new)

# new_2.describe()

# sns.boxplot(data = train, x = 'PoolQC', y = 'SalePrice')

# df.Temp_Rating.fillna(df.Farheit, inplace=True)

train2 = train

train2['PoolQC'].fillna('NA', inplace = True)

sns.boxplot(data = train2, x = 'PoolQC', y = 'SalePrice')



# pd.get_dummies(train['MiscFeature'])
