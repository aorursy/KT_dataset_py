# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression as lr

from sklearn.model_selection import cross_val_predict



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

sample = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
train = train.select_dtypes(include=['float','int'])

train_column = train.columns



test = train.select_dtypes(include=['float','int'])
train = train[train_column]

feature_names=train.columns.values


train = train[train_column]

figures_per_time = 2

count = 0 

y = train.SalePrice.values

for i in train_column:

    if i == 'Id' or i == 'SalePrice':

        continue

    x = train[i]

    plt.figure(count//figures_per_time,figsize = (25,10))

    plt.subplot(1,figures_per_time,np.mod(count,2)+1)

    plt.scatter(x, y);

    plt.title('f model: T= {}'.format(i))

    count+=1
test_col = test.columns

test_col
train_column
train = train.drop(['LotFrontage','MasVnrArea','GarageYrBlt'], axis=1)

test = test.drop(['LotFrontage','MasVnrArea','GarageYrBlt'], axis=1)

train_column = train.columns

train_column
from scipy import stats

from scipy.stats import norm, skew 

for i in train_column:

    if i == 'Id' or i == 'SalePrice':

        continue

    x = train[i]

    fig = plt.figure(figsize=(15,5))

    plt.subplot(1,2,1)

    sns.distplot(x , fit=norm);

    (mu, sigma) = norm.fit(x)

    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

                loc='best')

    plt.ylabel('Frequency')

    plt.title(i + ' distribution')

    plt.subplot(1,2,2)

    res = stats.probplot(x, plot=plt)

plt.suptitle('Before transformation')



for i in train_column:

    if i == 'Id' or i == 'SalePrice' or i == 'LotFrontage' or i == 'MasVnrArea' or i == 'GarageYrBlt':

        continue

    test[i] = np.log1p(test[i])

    x = test[i]

    fig = plt.figure(figsize=(15,5))

    plt.subplot(1,2,1)

    sns.distplot(x , fit=norm);

    (mu, sigma) = norm.fit(x)

    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

                loc='best')

    plt.ylabel('Frequency')

    plt.title(i + ' distribution')

    plt.subplot(1,2,2)

    res = stats.probplot(x, plot=plt)

plt.suptitle('After transformation')
reg = lr(normalize = True)#.fit(test, np.log1p(train['SalePrice']))

predictSale = cross_val_predict(reg,test, np.log1p(train['SalePrice']), cv=10)#reg.predict(test)



predictSale
matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)

sales = pd.DataFrame({"sale_p":predictSale,"sale_o":np.log1p(train['SalePrice'])})

sales.plot()
sum = 0

for i in range(len(predictSale-1)-1):

    sum = sum + (predictSale[i] - np.log1p(train['SalePrice'])[i])** 2

rmse = np.sqrt(sum/len(predictSale))

print(rmse)