# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot  as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

pd.options.display.max_rows = 1000

pd.options.display.max_columns = 20

%matplotlib inline



import os

print(os.listdir("../input"))

%config InlineBackend.figure_format ='retina'





# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
print("train shape "+ str(train.shape))

print("test shape " + str(test.shape))

quantitative = [f for f in train.columns if train.dtypes[f] != 'object']

quantitative.remove('SalePrice')

quantitative.remove('Id')

qualitative = [f for f in train.columns if train.dtypes[f] == 'object']
missing = train.isnull().sum()

missing = missing[missing > 0]

# print("number of attributes having missing values " + str(len(missing)))

missing.sort_values(inplace=True)

missing.plot.bar()
missing = train.isnull().sum()

missing = missing[missing > len(train)*0.05]

# print("number of attributes having missing values greter than 5% " + str(len(missing)))

missing.sort_values(inplace=True)

missing.plot.bar()
len(train)
import scipy.stats as st

y = train['SalePrice']

plt.figure(1); plt.title('Normal')

sns.distplot(y, kde=True, fit=st.norm)

plt.figure(2); plt.title('Log Normal')

sns.distplot(y, kde=True, fit=st.lognorm)

plt.figure(3); plt.title('Johnson SB')

sns.distplot(y, kde=True, fit=st.johnsonsb)

plt.figure(4); plt.title('Johnson SU')

sns.distplot(y, kde=True, fit=st.johnsonsu)
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], size = 3)

plt.show();
for c in qualitative:

    train[c] = train[c].astype('category')

    if train[c].isnull().any():

        train[c] = train[c].cat.add_categories(['MISSING'])

        train[c] = train[c].fillna('MISSING')



def boxplot(x, y, **kwargs):

    sns.boxplot(x=x, y=y)

    x=plt.xticks(rotation=90)

f = pd.melt(train, id_vars=['SalePrice'], value_vars=qualitative)

g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)

g = g.map(boxplot, "value", "SalePrice")