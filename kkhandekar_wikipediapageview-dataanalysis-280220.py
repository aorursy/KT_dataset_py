#importing libaries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from datetime import datetime
#loading dataset

data = pd.read_csv('../input/wikipediapageviewschristmas/pageviews-20180226-20200226.csv',header='infer')
data.shape
#Checking for null/missing values

data.isna().sum()
#Checking for column data types

data.dtypes
data.head()
plt.figure(figsize=(15,6))

plt.title('Distribution of Views per day', fontsize=16)

plt.tick_params(labelsize=14)

sns.distplot(data['Christmas'], bins=60);
#Converting date to Pandas DateTime

data['Converted_Date'] = pd.to_datetime(data['Date'])
# add column 'Day', 'Month', 'Year' to the dataframe

data['Day'] = data['Converted_Date'].dt.day

data['Month'] = data['Converted_Date'].dt.month

data['Year'] = data['Converted_Date'].dt.year
#Converting the date column to index

data.index = pd.DatetimeIndex(data['Date'])

data = data.drop(columns=['Date','Converted_Date'],axis=1)
data.head()
view_pivot = pd.pivot_table(data, values='Christmas', index=['Month'],

                    columns=['Year'])



view_pivot.plot(figsize=(10,8))
sns.set(rc={'figure.figsize':(11, 4)})

data['Christmas'].plot(linewidth=0.5);
ax = data.loc['2019':'2020', 'Christmas'].plot(marker='o', linestyle='-')

ax.set_ylabel('Views');