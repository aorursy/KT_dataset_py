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
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

from pandas.plotting import lag_plot, autocorrelation_plot

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.stattools import adfuller

 

%matplotlib inline
df = pd.read_excel('/kaggle/input/pm-25/PM25_1g.xlsx', index_col=0)
df.head()
df.shape
df.dtypes
cols = ['Niepodleglosci','Wokalna']

df[cols] = df[cols].replace(',','.', regex=True).astype(float)
df.dtypes
df.isna().sum()
df['Niepodleglosci'].fillna(df['Niepodleglosci'].mean(), inplace=True)

 

df['Wokalna'].fillna(df['Wokalna'].mean(), inplace=True)
df.isna().sum()
df.describe()
df['Niepodleglosci'].plot()

plt.xlabel('Date')

plt.ylabel(cols[0])

plt.legend()

plt.show()
df['Wokalna'].plot()

plt.xlabel('Date')

plt.ylabel(cols[0])

plt.legend()

plt.show()
plt.figure(figsize=(15, 6))

df.resample('M').mean().plot(figsize=(15, 6));



df.boxplot(figsize=(10, 6))

plt.show()
monthly = df.resample('d').mean()

monthly['month'] = monthly.index.strftime('%B')

 

 

f_box = plt.figure(figsize=(15, 16))

f_box.subplots_adjust(hspace=0.4)

for i in range(2):

    ax = f_box.add_subplot(4, 1, i+1)

    ax = sns.boxplot(data=monthly, x=monthly['month'], y=monthly.iloc[:, i])
df.hist(figsize=(10, 6))

plt.show()
lag_plot(df['Niepodleglosci'], lag=1)

plt.title('Niepodleglosci')
lag_plot(df['Wokalna'], lag=1)

plt.title('Wokalna')
#Niepodloeglosci

decomp = seasonal_decompose(df['Niepodleglosci'], model='multiplicative')

decomp.plot()

plt.show()
#Wokalna

decomp1 = seasonal_decompose(df['Wokalna'], model='multiplicative')

decomp1.plot()

plt.show()
#Niepodleglosci



X = df['Niepodleglosci'].values

result = adfuller(X)

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():

	print('\t%s: %.3f' % (key, value))
#Wokalna



X = df['Wokalna'].values

result = adfuller(X)

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():

	print('\t%s: %.3f' % (key, value))