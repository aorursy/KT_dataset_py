#importing libaries

import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import scipy.stats as st

from sklearn import ensemble, tree, linear_model

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
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test.describe()
train.describe()
train.head()
print(train.shape)

print(test.shape)
numeric=[f for f in train.columns if train.dtypes[f] != 'object']

numeric
catagorical = [i for i in train.columns if train.dtypes[i] == 'object']

catagorical
num_features = train.select_dtypes(include=[np.number])

num_features.columns
cat_features = train.select_dtypes(include=[object])

cat_features.columns
numeric_features=num_features.drop(['YrSold','YearBuilt', 'YearRemodAdd', 'GarageYrBlt'],axis=1)
numeric_features
discrete_feature=[feature for feature in numeric_features if len(train[feature].unique())<25]

discrete_feature
for feature in discrete_feature:

    data=train.copy()

    data.groupby(feature)['SalePrice'].median().plot.bar()

    plt.xlabel(feature)

    plt.ylabel('SalePrice')

    plt.title(feature)

    plt.show()
continuous_feature=[feature for feature in numeric_features if feature not in discrete_feature]

continuous_feature.pop(0)
for feature in continuous_feature:

    data=train.copy()

    data[feature].hist(bins=25)

    plt.xlabel(feature)

    plt.ylabel('SalePrice')

    plt.title(feature)

    plt.show()
train['HouseStyle'].unique() 

for feature in cat_features:

    

    sns.countplot(x=feature, data=cat_features);

    plt.title(feature)

    plt.show()

train.info()

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1,join='outer', keys=['Total Missing Count', ' % of Total Observations'])

missing_data.index.name ='Feature'

missing_data.head(20)

missing = train.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False)

for feature in continuous_feature:

    data=train.copy()

    if 0 in data[feature].unique():

        pass

    else:

        data[feature]=np.log(data[feature])

        data.boxplot(column=feature)

        plt.ylabel(feature)

        plt.title(feature)

        plt.show()
train.skew()
train.kurt()

import scipy.stats as stats



y = train['SalePrice']

plt.figure(1); plt.title('Johnson SU')

sns.distplot(y, kde=False, fit=stats.johnsonsu)

plt.figure(2); plt.title('Normal')

sns.distplot(y, kde=False, fit=stats.norm)

plt.figure(3); plt.title('Log Normal')

sns.distplot(y, kde=False, fit=stats.lognorm)
corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True)
k= 11

cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index

print(cols)

cm = np.corrcoef(train[cols].values.T)

f , ax = plt.subplots(figsize = (14,12))

sns.heatmap(cm, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',

            linecolor="white",xticklabels = cols.values ,annot_kws = {'size':12},yticklabels = cols.values)