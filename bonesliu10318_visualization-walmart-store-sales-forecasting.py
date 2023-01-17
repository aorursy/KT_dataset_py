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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

from scipy import stats

from scipy.stats import norm

import seaborn as sns

import os

import warnings

warnings.filterwarnings('ignore')



sns.set_style('whitegrid')

%matplotlib inline
path = '/kaggle/input/walmart-recruiting-store-sales-forecasting/'



features = pd.read_csv(path + 'features.csv.zip')

sampleSubmission = pd.read_csv(path + 'sampleSubmission.csv.zip')

stores = pd.read_csv(path + 'stores.csv')

train = pd.read_csv(path + 'train.csv.zip')

test = pd.read_csv(path + 'test.csv.zip')
sampleSubmission.sample(4)
print(features.shape)

features.sample(4)
print(train.shape)

train.sample(4)
print(test.shape)

test.sample(4)
print(stores.shape)

stores.sample(4)
data_val = test.copy(deep = True)

data_train = train.copy(deep = True)
import pandas as pd

stores = pd.read_csv("../input/walmart-recruiting-store-sales-forecasting/stores.csv")
import pandas as pd

stores = pd.read_csv("../input/walmart-recruiting-store-sales-forecasting/stores.csv")
combined_train = pd.merge(data_train, stores, how = 'left', on = 'Store')

combined_test = pd.merge(data_val, stores, how = 'left', on = 'Store')
combined_train = pd.merge(combined_train, features, how = 'inner', on = ['Store', 'Date'])

combined_test = pd.merge(combined_test, features, how = 'inner', on = ['Store', 'Date'])
data = pd.concat([combined_train, combined_test], axis = 0, ignore_index = True)
print('The size of the train data {}'.format(train.shape[0]))
# Make datetypes constant for all datasets



data['Date'] = pd.to_datetime(data['Date'], format="%Y-%m-%d")
data.drop(['IsHoliday_y'], axis = 1, inplace = True)

data.rename(columns = {'IsHoliday_x': 'IsHoliday'}, inplace = True)
Total = data.isnull().sum().sort_values(ascending = False)

Percent = Total / data.shape[0]

DataTypes = data.dtypes



pd.concat([Total, Percent, DataTypes], axis = 1, keys = ['Total', 'Percent', 'DataTypes']).sort_values(by = ['Total'], ascending = False)
data.loc[data['Weekly_Sales'] < 0, 'Weekly_Sales'] = 0

data.loc[data['MarkDown1'] < 0, 'Weekly_Sales'] = 0

data.loc[data['MarkDown2'] < 0, 'Weekly_Sales'] = 0

data.loc[data['MarkDown3'] < 0, 'Weekly_Sales'] = 0

data.loc[data['MarkDown4'] < 0, 'Weekly_Sales'] = 0

data.loc[data['MarkDown5'] < 0, 'Weekly_Sales'] = 0
data.fillna(0, inplace = True)

data.isnull().sum().to_frame()
data.dtypes
plt.figure(figsize = (12, 10))



sns.set_style('whitegrid')



ax0 = sns.lineplot(x = 'Date', y = 'CPI', data = data[0:421570].loc[data['CPI'] < 180], palette = 'mako')

ax0.set(title = 'The CPI of the City ', xlabel = 'Date', ylabel = 'CPI index')
plt.figure(figsize = (12, 10))



sns.set_style('whitegrid')



ax1 = sns.lineplot(x = 'CPI', y = 'Weekly_Sales', data = data[0:421570].loc[data['CPI'] < 180], palette = 'mako')

plt.title('The relationship of CPI and Weekly Sales', fontsize = 15)

plt.xlabel('CPI Index', fontsize = 15)

plt.ylabel('Weekly Sales', fontsize = 15)
plt.figure(figsize = (12, 10))



sns.set_style('whitegrid')



ax2 = sns.catplot(x = 'Size', y = 'Weekly_Sales', data = data.loc[0:421570], palette = 'mako')

# ax2.set(title = 'The relationship of Size and Weekly Sales', xlabel = 'Size', ylabel = 'Weekly Sales')

plt.title('The relationship of Size and Weekly Sales', fontsize = 15)

plt.xlabel('Size', fontsize = 15)

plt.ylabel('Weekly Sales', fontsize = 15)

plt.show()
plt.figure(figsize = (12, 10))



sns.set_style('whitegrid')



ax3 = sns.boxplot(y = 'Weekly_Sales', x = 'Type', hue = 'IsHoliday', data = data.loc[0:421570].loc[data['Weekly_Sales'] < 50000], palette = 'mako')

ax3.set(title = 'The relationship of Weekly Sales and Stroe Types', xlabel = 'Store Type', ylabel = 'Week Sales')
plt.figure(figsize = (12, 10))



sns.set_style('whitegrid')



ax4 = sns.violinplot(x = 'Type', y = 'Weekly_Sales', hue = 'IsHoliday', split = True, data = data.loc[0:421570].loc[data['Weekly_Sales'] < 100000], palette = 'mako')

ax4.set(title = 'The distrubtion of the Type and Weekly Sales', xlabel = 'The type of the Stores', ylabel = 'Weekly Sales')
plt.figure(figsize = (12, 10))

sns.set_style('whitegrid')

ax5 = sns.catplot(x = 'Size', hue = 'Type', kind = 'count', data = data.loc[0:421570], palette = 'mako')

ax5.set(title = 'The distrubtion of the Type of the stores', xlabel = 'The size of the Stores', ylabel = 'Type of the stores')

plt.xticks(rotation = 45)
plt.figure(figsize = (12, 10))



sns.set_style('whitegrid')



ax6 = sns.distplot(data.loc[0:421570, 'Weekly_Sales'], hist = True, fit = norm)
ax7 = plt.figure(figsize = (12, 10))



ax7.add_subplot(1, 2, 1)

res = stats.probplot(data.loc[0:421570, 'Weekly_Sales'] ,  plot = plt)



ax7.add_subplot(1, 2, 2)

res = stats.probplot(np.log1p(data.loc[0:421570, 'Weekly_Sales']) , plot = plt)
groupedData = data.loc[0:421570].groupby(['Dept', 'Date']).mean().round(0).reset_index()

print(groupedData.shape)

groupedData.sample(4)
print('The values of Dept'.center(50, '-'))

print(data.loc[0:421570,'Dept'].unique())

print('The number of Dept'.center(50, '-'))

data.loc[0:421570, 'Dept'].nunique()
print('The values of Dept'.center(50, '-'))

print(groupedData['Dept'].unique())

print('The number of Dept'.center(50, '-'))

groupedData['Dept'].nunique()
groupedData2 = groupedData[['Dept', 'Date', 'Weekly_Sales']]

groupedData2.sample(4)
dept = groupedData2['Dept'].unique()



dept.sort()

dept_1=dept[0:20]

dept_2=dept[20:40]

dept_3=dept[40:60]

dept_4=dept[60:]



fig, ax = plt.subplots(2,2,figsize=(20,10))

fig.subplots_adjust(wspace=0.5, hspace=0.5)

fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)



for i in dept_1 :

    data_1=groupedData2[groupedData2['Dept']==i]

    ax[0,0].plot(data_1['Date'], data_1['Weekly_Sales'],label='Dept_1_mean_sales')



for i in dept_2 :

    data_1=groupedData2[groupedData2['Dept']==i]

    ax[0,1].plot(data_1['Date'], data_1['Weekly_Sales'],label='Dept_1_mean_sales')

    

for i in dept_3 :

    data_1=groupedData2[groupedData2['Dept']==i]

    ax[1,0].plot(data_1['Date'], data_1['Weekly_Sales'],label='Dept_1_mean_sales')    



for i in dept_4 :

    data_1=groupedData2[groupedData2['Dept']==i]

    ax[1,1].plot(data_1['Date'], data_1['Weekly_Sales'],label='Dept_1_mean_sales')        

    

ax[0,0].set_title('Mean sales record by department(0~19)')

ax[0,1].set_title('Mean sales record by department(20~39)')

ax[1,0].set_title('Mean sales record by department(40~59)')

ax[1,1].set_title('Mean sales record by department(60~)')





ax[0,0].set_ylabel('Mean sales')

ax[0,0].set_xlabel('Date')

ax[0,1].set_ylabel('Mean sales')

ax[0,1].set_xlabel('Date')

ax[1,0].set_ylabel('Mean sales')

ax[1,0].set_xlabel('Date')

ax[1,1].set_ylabel('Mean sales')

ax[1,1].set_xlabel('Date')





plt.show()
groupedData3 = data.loc[0:421570].groupby(['Store', 'Date']).mean().reset_index()

groupedData3.shape
print(groupedData3['Store'].unique())

print(groupedData3['Store'].nunique())
groupedData4 = groupedData3[['Store', 'Date', 'Weekly_Sales']]

groupedData4.head(4)
store = groupedData4['Store'].unique()
store_1 = store[0 : 10]

store_2 = store[11 : 21]

store_3 = store[22 : 33]

store_4 = store[34 : -1]



fig, ax = plt.subplots(2,2,figsize=(20,10))

fig.subplots_adjust(wspace=0.5, hspace=0.5)

fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)



for i in store_1 :

    data_1=groupedData4[groupedData4['Store']==i]

    ax[0,0].plot(data_1['Date'], data_1['Weekly_Sales'],label='Dept_1_mean_sales')



for i in store_2 :

    data_1=groupedData4[groupedData4['Store']==i]

    ax[0,1].plot(data_1['Date'], data_1['Weekly_Sales'],label='Dept_1_mean_sales')

    

for i in store_3 :

    data_1=groupedData4[groupedData4['Store']==i]

    ax[1,0].plot(data_1['Date'], data_1['Weekly_Sales'],label='Dept_1_mean_sales')    



for i in store_4 :

    data_1=groupedData4[groupedData4['Store']==i]

    ax[1,1].plot(data_1['Date'], data_1['Weekly_Sales'],label='Dept_1_mean_sales')        

    

ax[0,0].set_title('Mean sales record by department(0~10)')

ax[0,1].set_title('Mean sales record by department(11~21)')

ax[1,0].set_title('Mean sales record by department(22~33)')

ax[1,1].set_title('Mean sales record by department(34~)')





ax[0,0].set_ylabel('Mean sales')

ax[0,0].set_xlabel('Date')

ax[0,1].set_ylabel('Mean sales')

ax[0,1].set_xlabel('Date')

ax[1,0].set_ylabel('Mean sales')

ax[1,0].set_xlabel('Date')

ax[1,1].set_ylabel('Mean sales')

ax[1,1].set_xlabel('Date')





plt.show()
print(data.shape)

data.head(4)