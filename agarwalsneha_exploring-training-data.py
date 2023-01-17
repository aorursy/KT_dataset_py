# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Load data

train = pd.read_csv('../input/train.csv')

#sneakpeak

train.head()
train.describe()
price = train['SalePrice']

plt.hist(price, bins='auto')
#We have some outliers. We might want to discard them.

#train = train[train['SalePrice'] < 5*train['SalePrice'].std()]

#price = train['SalePrice']

#plt.hist(price, bins='auto')
train.isnull().sum()[train.isnull().sum()!=0]
#train['PoolQC'].dropna() #['Fa', 'Ex', 'Gd']

#train['MiscFeature'].dropna() #['Shed', 'Othr']

#train['Alley'].dropna() #['Grvl', 'Pave']

#good to know what they have. doesn't look useful at the moment, so we can drop them.



pricewAlley = train[train['Alley'].notnull()]['SalePrice']

plt.hist(pricewAlley, bins='auto')
pricewPoolQC = train[train['PoolQC'].notnull()]['SalePrice']

plt.hist(pricewPoolQC, bins='auto')
pricewMiscFeature = train[train['MiscFeature'].notnull()]['SalePrice']

plt.hist(pricewMiscFeature, bins='auto')
train.drop(['Alley','MiscFeature','PoolQC'],inplace=True,axis=1)
train.dtypes
#train[(train.dtypes == np.float64) & train.dtypes == np.int64]