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

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn import metrics

from scipy import stats
data = pd.read_csv('/kaggle/input/rk-puram-ambient-air/trainset.csv', index_col=0, header=0)

data.info()
f, axes = plt.subplots(1,4, figsize=(16,8))

sns.boxplot(x=data['AT'].values,ax=axes[0], orient='v').set(xlabel='AT', ylabel='Centigrade') #dataset box plot for AT

sns.boxplot(x=data['BP'].values,ax=axes[1], orient='v').set(xlabel='BP') #dataset box plot for BP

sns.boxplot(x=data['RH'].values,ax=axes[2], orient='v').set(xlabel='RH') #dataset box plot for RH

sns.boxplot(x=data['SR'].values,ax=axes[3], orient='v').set(xlabel='SR') #dataset box plot for SR

plt.show()
f, axes = plt.subplots(1,4, figsize=(16,8))

sns.boxplot(x=data['WD'].values,ax=axes[0], orient='v').set(xlabel='WD') #dataset box plot for WD

sns.boxplot(x=data['WS'].values,ax=axes[1], orient='v').set(xlabel='WS') #dataset box plot for WS

sns.boxplot(x=data['CO'].values,ax=axes[2], orient='v').set(xlabel='CO') #dataset box plot for CO

sns.boxplot(x=data['NH3'].values,ax=axes[3], orient='v').set(xlabel='NH3') #dataset box plot for NH3

plt.show()
f, axes = plt.subplots(1,4, figsize=(16,8))

sns.boxplot(x=data['NO2'].values,ax=axes[0], orient='v').set(xlabel='NO2') #dataset box plot for NO2

sns.boxplot(x=data['NO'].values,ax=axes[1], orient='v').set(xlabel='NO') #dataset box plot for NO

sns.boxplot(x=data['NOx'].values,ax=axes[2], orient='v').set(xlabel='CO') #dataset box plot for NOx

sns.boxplot(x=data['O'].values,ax=axes[3], orient='v').set(xlabel='O') #dataset box plot for O

plt.show()
f, axes = plt.subplots(1,4, figsize=(16,8))

sns.boxplot(x=data['Ozone'].values,ax=axes[0], orient='v').set(xlabel='Ozone') #dataset box plot for Ozone

sns.boxplot(x=data['SO2'].values,ax=axes[1], orient='v').set(xlabel='SO2') #dataset box plot for SO2

sns.boxplot(x=data['PM25'].values,ax=axes[2], orient='v').set(xlabel='PM25') #dataset box plot for PM2.5

sns.boxplot(x=data['PM10'].values,ax=axes[3], orient='v').set(xlabel='PM10') #dataset box plot for PM10

plt.show()
data = data.drop(['Year', 'Month', 'Weekday', 'Hour'], axis=1)

data.shape
#IQL

Q1 = data.quantile(0.25)

Q3 = data.quantile(0.75)

IQR = Q3 - Q1

#print(IQR)

#print((data[:10000] < (Q1 - 1.5 * IQR)) |(data[:10000] > (Q3 + 1.5 * IQR)))

data_iqr = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]

print(data_iqr.shape) #12064 samples remaining after removing outliers by iqr method

#data_iqr.to_csv('train_iqr.csv', index=True)
#Z-score

z = np.abs(stats.zscore(data))

#print(z)

threshold = 3

#print(np.where(z > 3))

data_z = data[(z < 3).all(axis=1)]

#print(data[:10000].shape, data_z.shape)

print(data_z.shape)

#data_z.to_csv('train_z.csv', index=True)