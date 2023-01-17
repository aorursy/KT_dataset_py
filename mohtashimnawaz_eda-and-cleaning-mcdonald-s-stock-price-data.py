#Importing the libraries -- These are most basic libraries, others will be imported as needed

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/mcdonalds-stock-prices-20152020/mcdonalds-stocks-data.csv')

print("Shape of data:", data.shape)
# Let's see how data looks

data.head()
# I suspect some columns do not have the appropriate types. Let's check out

data.dtypes
data.columns
# If you look closely columns names contain an extra space, its annoying so we remove it.

data.columns = data.columns.str.strip()

data.columns
# $ signs make our work difficult, let's remove these dollar signs from entries

data['Low']=data['Low'].str.replace('$','').astype(float)

data['High']=data['High'].str.replace('$','').astype(float)

data['Open']=data['Open'].str.replace('$','').astype(float)

data['Close/Last']=data['Close/Last'].str.replace('$','').astype(float)
# Let's check if work is done

data.head()
# As suspected, its string type, I will convert to date type

# And create day, month and year column

dates = data['Date'].str.split('/')

days = dates.map(lambda x:x[1])

months = dates.map(lambda x:x[0])

years = dates.map(lambda x:x[2])

data['day']=days

data['month']=months

data['year']=years
# Lets see if new columns are added

data.head()
# I won't drop the date column as it may be relevent for exploration.

# I don't want to change the original dataset

# So I will convert datatype of date

data['Date'] = pd.to_datetime(data['Date'])
data.head()
# As we can see pandas is smart enough to handle date 

# Now let's check out final datatypes 

data.dtypes
sns.heatmap(data.isnull())
# Heatmap shows there are a few if not none, null places

# Let's see number of null values in each column

data.isnull().sum()
# Hurray, we don't have any null values

# Let's see distributions

# Let't begin with 'Volume'

sns.distplot(data['Volume'])

print("Skew:",data['Volume'].skew())

print("Kurtosis:", data["Volume"].kurtosis())
# The 'Volume' is right-skewed and it should be dealt with accordingly. One way is to use logarithm.

sns.distplot(np.log1p(data['Volume']))
# Let's check other attributes

plt.figure(figsize=(20,10))

ax1 = plt.subplot(2,2,1)

ax2=plt.subplot(2,2,2)

ax3=plt.subplot(2,2,3)

ax4=plt.subplot(2,2,4)

sns.distplot(data['Open'],ax=ax1)

sns.distplot(data['High'],ax=ax2)

sns.distplot(data['Low'],ax=ax3)

sns.distplot(data['Close/Last'],ax=ax4)
# Let's plot a box plot of volume

sns.boxplot(y='Volume', data=data)

plt.show()
# This was actually expected because there can be some abnormal changes in stocks

# Also it shows skewness of 'Volume' which we have seen above

# Let's plot for others

plt.figure(figsize=(20,10))

ax1 = plt.subplot(2,2,1)

ax2=plt.subplot(2,2,2)

ax3=plt.subplot(2,2,3)

ax4=plt.subplot(2,2,4)

sns.boxplot(y='Open',data=data,ax=ax1)

sns.boxplot(y='High',data=data,ax=ax2)

sns.boxplot(y='Low',data=data,ax=ax3)

sns.boxplot(y='Close/Last',data=data,ax=ax4)

plt.show()
# Let's plot volume-date because that is something very important

plt.figure(figsize=(10,6))

ax1=plt.subplot(1,2,1)

ax2=plt.subplot(1,2,2)

sns.lineplot(data['Date'],data['Volume'],ax=ax1)

sns.scatterplot(data['Date'],data['Volume'],ax=ax2)
# We shall plot scatter plots for other fields as well

plt.figure(figsize=(20,10))

ax1 = plt.subplot(2,2,1)

ax2=plt.subplot(2,2,2)

ax3=plt.subplot(2,2,3)

ax4=plt.subplot(2,2,4)

sns.scatterplot(y=data['Open'],x=data['Date'],ax=ax1)

sns.scatterplot(y=data['High'],x=data['Date'],ax=ax2)

sns.scatterplot(y=data['Low'],x=data['Date'],ax=ax3)

sns.scatterplot(y=data['Close/Last'],x=data['Date'],ax=ax4)

plt.show()
# There is something going on in 2020, let's see this (as if we don't know)

plt.figure(figsize=(20,10))

ax1 = plt.subplot(2,2,1)

ax2=plt.subplot(2,2,2)

ax3=plt.subplot(2,2,3)

ax4=plt.subplot(2,2,4)

data_for_2020 = data[data['year']=='2020']

sns.barplot(y=data_for_2020['Open'] ,x=data_for_2020['month'],ax=ax1)

sns.barplot(y=data_for_2020['High'],x=data_for_2020['month'],ax=ax2)

sns.barplot(y=data_for_2020['Low'],x=data_for_2020['month'],ax=ax3)

sns.barplot(y=data_for_2020['Close/Last'],x=data_for_2020['month'],ax=ax4)

plt.show()
# Correlation plot

sns.heatmap(data.corr())
# Let's see relations among open-low,open-high,close-low,close-high

plt.figure(figsize=(20,10))

ax1 = plt.subplot(2,2,1)

ax2=plt.subplot(2,2,2)

ax3=plt.subplot(2,2,3)

ax4=plt.subplot(2,2,4)

sns.scatterplot(y=data['Open'],x=data['Low'],ax=ax1)

sns.scatterplot(y=data['Open'],x=data['High'],ax=ax2)

sns.scatterplot(y=data['Close/Last'],x=data['Low'],ax=ax3)

sns.scatterplot(y=data['Close/Last'],x=data['High'],ax=ax4)

plt.show()
# Let's see some scatter plots between low and high, we shall see a high correlation

sns.scatterplot(x=data['Low'],y=data['High'])
# At last let's plot volume vs low, high, close/last and open

plt.figure(figsize=(20,10))

ax1 = plt.subplot(2,2,1)

ax2=plt.subplot(2,2,2)

ax3=plt.subplot(2,2,3)

ax4=plt.subplot(2,2,4)

sns.scatterplot(y=data['Volume'],x=data['Low'],ax=ax1)

sns.scatterplot(y=data['Volume'],x=data['High'],ax=ax2)

sns.scatterplot(y=data['Volume'],x=data['Close/Last'],ax=ax3)

sns.scatterplot(y=data['Volume'],x=data['Open'],ax=ax4)

plt.show()
# Let's filter the outliers and save the new dataframe to a new file

data = data[data['Volume']<2.0e7]

print(data.shape)

data.to_csv('final.csv')