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
traindata = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
traindata
testdata = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
testdata
newdata = pd.merge(traindata, testdata)
newdata.head()
class display(object):

    template = """<div style="float: left; padding: 10px;">

    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}

    </div>"""

    def __init__(self, *args):

        self.args = args

        

    def _repr_html_(self):

        return '\n'.join(self.template.format(a, eval(a)._repr_html_())

                         for a in self.args)

    

    def __repr__(self):

        return '\n\n'.join(a + '\n' + repr(eval(a))

                           for a in self.args)
df1 = pd.DataFrame(traindata)

df2 = pd.DataFrame(testdata)

display('traindata', 'testdata')
df3 = pd.merge(df1, df2)
df1.columns
df2.columns
df1['SalePrice'].describe()
import seaborn as sns

sns.distplot(df1['SalePrice'])
var = 'GrLivArea'

data = pd.concat([df1['SalePrice'], df1[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
var = 'TotalBsmtSF'

data = pd.concat([df1['SalePrice'], df1[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
var = 'OverallQual'

data = pd.concat([df1['SalePrice'], df1[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
import matplotlib.pyplot as plt

var = 'YearBuilt'

data = pd.concat([df1['SalePrice'], df1[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
corrmat = df1.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
k = 10

cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index

cm = np.corrcoef(df1[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df1[cols], height = 2.5)

plt.show()
total = df1.isnull().sum().sort_values(ascending=False)

percent = (df1.isnull().sum()/df1.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
saleprice_scaled = StandardScaler().fit_transform(df1['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
from sklearn.preprocessing import StandardScaler

var = 'GrLivArea'

data = pd.concat([df1['SalePrice'], df1[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
df1.sort_values(by = 'GrLivArea', ascending = False)[:2]

df1 = df1.drop(df1[df1['Id'] == 1299].index)

df1 = df1.drop(df1[df1['Id'] == 524].index)
var = 'TotalBsmtSF'

data = pd.concat([df1['SalePrice'], df1[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
sns.distplot(df1['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(df1['SalePrice'], plot=plt)
from scipy.stats import norm

from scipy import stats

df1['SalePrice'] = np.log(df1['SalePrice'])
sns.distplot(df1['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(df1['SalePrice'], plot=plt)
sns.distplot(df1['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(df1['GrLivArea'], plot=plt)
df1['GrLivArea'] = np.log(df1['GrLivArea'])

sns.distplot(df1['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(df1['GrLivArea'], plot=plt)
sns.distplot(df1['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(df1['TotalBsmtSF'], plot=plt)
df1.loc[df1['TotalBsmtSF']==1,'TotalBsmtSF'] = np.log(df1['TotalBsmtSF'])
df1['HasBsmt'] = pd.Series(len(df1['TotalBsmtSF']), index=df1.index)

df1['HasBsmt'] = 0 

df1.loc[df1['TotalBsmtSF']>0,'HasBsmt'] = 1
df1.loc[df1['TotalBsmtSF']==1,'TotalBsmtSF'] = np.log(df1['TotalBsmtSF'])
plt.scatter(df1['GrLivArea'], df1['SalePrice']);

plt.scatter(df1[df1['TotalBsmtSF']>0]['TotalBsmtSF'], df1[df1['TotalBsmtSF']>0]['SalePrice']);
df1 = pd.get_dummies(df1)