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
# text file(.txt) converts into excel(CSV) file(.csv) and create csv file with rows and columns wise data

"""import csv

with open('C:/Users/Admin/Desktop/household_power_consumption.txt', 'r') as in_file:

    stripped = (line.strip() for line in in_file)

    lines = (line.split(";") for line in stripped if line)

    with open('household_power_consumption.csv', 'w') as out_file:

        writer = csv.writer(out_file)

        writer.writerows(lines)

"""
"""# read dataset

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

data1=pd.read_csv('household_power_consumption.csv')

data1.head()"""
import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

data=pd.read_csv('/kaggle/input/electric-power-consumption-data-set/household_power_consumption.txt', sep=';', 

                 parse_dates={'date_time' : ['Date', 'Time']}, infer_datetime_format=True, 

                 low_memory=False, na_values=['nan','?'], index_col='date_time')

data.head()
data.info()
data.shape # checking how much raws and columns in dataset 
#Counting the missing values

count=0

for i in data.isnull().sum(axis=1):

    if i>0:

        count=count+1

print('Total number of rows with missing values is ', count)

print('since it is only',round((count/len(data.index))*100), 'percent of the entire dataset the rows with missing values are excluded.')

data.describe()
data.isnull().sum()
# plot boxplot for checking mean and median values for filling null values.

# we know which are better for filling null values mean or median.

plt.figure(figsize=(15,8))

sns.boxplot(data=data.iloc[:,:2])
# plot boxplot for checking mean and median values for filling null values.

# we know which are better for filling null values mean or median.

plt.figure(figsize=(15,8))

sns.boxplot(data=data.iloc[:,2:4])
# plot boxplot for checking mean and median values for filling null values.

# we know which are better for filling null values mean or median.

plt.figure(figsize=(15,8))

sns.boxplot(data=data.iloc[:,4:])
# filling nan with mean in any columns

for i in range(0,7):        

        data.iloc[:,i]=data.iloc[:,i].fillna(data.iloc[:,i].mean())
# another sanity check to make sure that there are not more any nan

data.isnull().sum()
# Sum of 'Global_active_power' resampled over month

data['Global_active_power'].resample('M').sum().head()
# mean of 'Global_active_power' resampled over month

data['Global_active_power'].resample('M').mean().head()
plt.figure(figsize=(15,8)) # size of plot

# Sum of 'Global_active_power' resampled over month

data['Global_active_power'].resample('M').sum().plot(kind='bar')

plt.xticks(rotation=60) #sequence Values to use for the xticks

plt.ylabel('Global_active_power')

plt.title('sum of Global_active_power per month')

plt.show()

plt.figure(figsize=(15,8)) # size of plot

# Sum of 'Global_active_power' resampled over month

data['Global_active_power'].resample('M').sum().plot(kind='line')

plt.xticks(rotation=60)

plt.ylabel('Global_active_power')

plt.title('flow - sum of power consumptions Global_active_power ')

plt.show()
plt.figure(figsize=(15,8)) # size of plot

# Mean of 'Global_active_power' resampled over month

data['Global_active_power'].resample('M').mean().plot(kind='bar')

plt.xticks(rotation=60) #sequence Values to use for the xticks

plt.ylabel('Global_active_power')

plt.title('Global_active_power per month (averaged over month)')

plt.show()
plt.figure(figsize=(15,8)) # size of plot

# mean of 'Global_active_power' resampled over month

data['Global_active_power'].resample('M').mean().plot(kind='line')

plt.xticks(rotation=60)

plt.ylabel('Global_active_power')

plt.title('flow of consumptions Global_active_power (averaged over months)')

plt.show()
# Sum of 'Global_reactive_power' resampled over month

data['Global_reactive_power'].resample('M').sum().head()
plt.figure(figsize=(15,8))

# Sum of 'Global_reactive_power' resampled over month

data['Global_reactive_power'].resample('M').sum().plot(kind='bar')

plt.xticks(rotation=60)

plt.ylabel('Global_reactive_power')

plt.title('Sum of Global_reactive_power per month')

plt.show()
plt.figure(figsize=(15,8))

# Sum of 'Global_reactive_power' resampled over month

data['Global_reactive_power'].resample('M').sum().plot(kind='line')

plt.xticks(rotation=60)

plt.ylabel('Global_reactive_power')

plt.title('flow - sum of consumptions  Global_reactive_power')

plt.show()
plt.figure(figsize=(15,8))

# mean of 'Global_reactive_power' resampled over month

data['Global_reactive_power'].resample('M').mean().plot(kind='bar')

plt.xticks(rotation=60)

plt.ylabel('Global_reactive_power')

plt.title('Global_reactive_power per month (averaged over month)')

plt.show()
plt.figure(figsize=(15,8))

# mean of 'Global_reactive_power' resampled over month

data['Global_reactive_power'].resample('M').mean().plot(kind='line')

plt.xticks(rotation=60)

plt.ylabel('Global_reactive_power')

plt.title('flow of consumptions  Global_reactive_power(averaged over month)')

plt.show()
# Sum of 'Sub_metering_1' resampled over month

data['Sub_metering_1'].resample('M').sum().head()
plt.figure(figsize=(15,8))

# Sum of 'Sub_metering_1' resampled over month

data['Sub_metering_1'].resample('M').sum().plot(kind='bar')

plt.xticks(rotation=60)

plt.ylabel('Sub_metering_1')

plt.title('Sum of Sub_metering_1 per month')

plt.show()
plt.figure(figsize=(15,8))

# Sum of 'Sub_metering_1' resampled over month

data['Sub_metering_1'].resample('M').sum().plot(kind='line')

plt.xticks(rotation=60)

plt.ylabel('Sub_metering_1')

plt.title(' flow -sum  of power consumptions Sub_metering_1')

plt.show()
plt.figure(figsize=(15,8))

# mean of 'Sub_metering_1' resampled over month

data['Sub_metering_1'].resample('M').mean().plot(kind='bar')

plt.xticks(rotation=60)

plt.ylabel('Sub_metering_1')

plt.title('Sub_metering_1 per month (averaged over month)')

plt.show()
plt.figure(figsize=(15,8))

# mean of 'Sub_metering_1' resampled over month

data['Sub_metering_1'].resample('M').mean().plot(kind='line')

plt.xticks(rotation=60)

plt.ylabel('Sub_metering_1')

plt.title(' flow of power consumptions Sub_metering_1  (averaged over month)')

plt.show()
# Sum of 'Sub_metering_2' resampled over month

data['Sub_metering_2'].resample('M').sum().head()
plt.figure(figsize=(15,8))

# Sum of 'Sub_metering_2' resampled over month

data['Sub_metering_2'].resample('M').sum().plot(kind='bar')

plt.xticks(rotation=60)

plt.ylabel('Sub_metering_2')

plt.title('Sum of Sub_metering_2 per month ')

plt.show()

plt.figure(figsize=(15,8))

# Sum of 'Sub_metering_2' resampled over month

data['Sub_metering_2'].resample('M').sum().plot(kind='line')

plt.xticks(rotation=60)

plt.ylabel('Sub_metering_2')

plt.title('flow - sum of power consumptions Sub_metering_2 ')

plt.show()
plt.figure(figsize=(15,8))

# mean of 'Sub_metering_2' resampled over month

data['Sub_metering_2'].resample('M').mean().plot(kind='bar')

plt.xticks(rotation=60)

plt.ylabel('Sub_metering_2')

plt.title('Sub_metering_2 per month (averaged over month)')

plt.show()
plt.figure(figsize=(15,8))

# mean of 'Sub_metering_2' resampled over month

data['Sub_metering_2'].resample('M').mean().plot(kind='line')

plt.xticks(rotation=60)

plt.ylabel('Sub_metering_2')

plt.title('flow of power consumptions Sub_metering_2 (averaged over month)')

plt.show()
# Sum of 'Sub_metering_3' resampled over month

data['Sub_metering_3'].resample('M').sum().head()
plt.figure(figsize=(15,8))

# Sum of 'Sub_metering_3' resampled over month

data['Sub_metering_3'].resample('M').sum().plot(kind='bar')

plt.xticks(rotation=60)

plt.ylabel('Sub_metering_3')

plt.title('Sub_metering_3 per month ')

plt.show()
plt.figure(figsize=(15,8))

# Sum of 'Sub_metering_3' resampled over month

data['Sub_metering_3'].resample('M').sum().plot(kind='line')

plt.xticks(rotation=60)

plt.ylabel('Sub_metering_3')

plt.title('flow - sum of power consumptions  Sub_metering_3  ')

plt.show()
plt.figure(figsize=(15,8))

# mean of 'Sub_metering_3' resampled over month

data['Sub_metering_3'].resample('M').mean().plot(kind='bar')

plt.xticks(rotation=60)

plt.ylabel('Sub_metering_3')

plt.title('Sub_metering_3 per month (averaged over month)')

plt.show()
plt.figure(figsize=(15,8))

# Sum of 'Sub_metering_3' resampled over month

data['Sub_metering_3'].resample('M').mean().plot(kind='line')

plt.xticks(rotation=60)

plt.ylabel('Sub_metering_3')

plt.title('flow of power consumptions  Sub_metering_3  (averaged over month)')

plt.show()