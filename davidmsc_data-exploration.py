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



import matplotlib.pyplot as plt
## Get data

## Load data

data = pd.read_csv("/kaggle/input/demand-forecasting-kernels-only/train.csv")



## Convert datatypes

data['date'] = pd.to_datetime(data['date'])
data.describe()
print("Sales range from {} to {}.".format(data['sales'].min(), data['sales'].max()))

plt.hist(data['sales'], color='navy')

plt.title("Sales for all stores")

plt.xlabel("Sales")

plt.ylabel("Frequency")

plt.show()
data.dtypes
data.head()
data.isnull().sum()
print("Data from {} to {}".format(data['date'].min(), data['date'].max()))

print("Number of samples: {}".format(len(data)))

print("Stores: {}".format(data['store'].unique().tolist()))

print("Different items: {}".format(data['item'].unique()))
from pandas.plotting import register_matplotlib_converters



## Sales of one item from one store

plt.figure(figsize=(20,10))



for i in range(4):

    plt.subplot(2, 2, i+1)

    data_slice = data.loc[data['item'] == 3+i]

    data_slice = data_slice.loc[data_slice['store'] == 2+i]

    plt.plot(data_slice['date'], data_slice['sales'], color='navy')



plt.show()
pd.plotting.lag_plot(data['sales'])

plt.show()
plt.figure(figsize=[20,10])

pd.plotting.autocorrelation_plot(data.loc[data['store'] == int(3)].loc[data['item'] == int(39)]['sales'])

plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose

test = data.loc[data['store'] == int(3)].loc[data['item'] == int(39)]

test.index = test.date

test = test['sales']



decomposed = seasonal_decompose(test, model='additive')

x = decomposed.plot()
data_year = data.loc[(data['date'] >= '2015-01-01') & (data['date'] < '2016-01-01')]

data_year.loc[data_year['store'] == 8].loc[data_year['item'] == 35]['sales'].plot()
def moving_average(a, n=3) :

    ret = np.cumsum(a, dtype=float)

    ret[n:] = ret[n:] - ret[:-n]

    return ret[n - 1:] / n



plt.figure(figsize=(20,10))

weeks = [1, 4, 8, 12]

sales = data.loc[data['store'] == 8].loc[data['item'] == 35]['sales'].tolist()



for i in range(4):

    plt.subplot(2, 2, i+1)

    res = moving_average(sales, n=7*(weeks[i]))

    plt.plot(res)

    

plt.show()
data_month = data_year.loc[(data_year['date'] >= '2015-05-01') & (data_year['date'] < '2015-06-01')]

data_month.loc[data_month['store'] == 8].loc[data_month['item'] == 35]['sales'].plot()
test = data_month.loc[data_month['store'] == int(3)].loc[data_month['item'] == int(39)]

test.index = test.date

test = test['sales']



decomposed = seasonal_decompose(test, model='additive')

x = decomposed.plot()