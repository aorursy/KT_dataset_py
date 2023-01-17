# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
# reading the data from the csv files

data = pd.read_csv('../input/forest-fires-in-brazil/amazon.csv', encoding='latin1')
# displaying first 5 data from the dataframes

data.head()
# total length of data

len(data)
data.isna().sum()
# checking out the unique state data in the pandas dataframe

data.state.unique()
# checking out the unique month data in dataframe

data.month.unique()
# it's been clearly seen that no entry is null in our dataset

data.info()
#creating a dictionary with translations of months

month_map={'Janeiro': 'January', 'Fevereiro': 'February', 'Março': 'March', 'Abril': 'April', 'Maio': 'May',

          'Junho': 'June', 'Julho': 'July', 'Agosto': 'August', 'Setembro': 'September', 'Outubro': 'October',

          'Novembro': 'November', 'Dezembro': 'December'}

#mapping our translated months

data['month']=data['month'].map(month_map)

#checking the month column for the second time after the changes were made

data.month.unique()
# checking out the percentile of the number

data.number.describe()
data.head()
data.year.unique()
data.number.sum()
plt.figure(figsize=(20,10))

sns.swarmplot(x = 'month', y = 'number',data = data)

plt.xticks(rotation = 90)

plt.show()
plt.figure(figsize = (20,5))

sns.violinplot(x='month',y='number',data = data)

plt.show()
data.boxplot(column="number", by="year")

plt.xticks(rotation=75)

plt.show()
# finding out the unique years 

data['Year'] = pd.DatetimeIndex(data['date']).year
plt.figure(figsize = (20,5))

sns.lineplot(x="Year", y="number", data=data)

plt.xticks(rotation=15)

plt.title('seaborn-matplotlib example')

plt.show()