import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sn

%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib

matplotlib.style.use('ggplot')

from datetime import datetime
#Reading of the data 

data = pd.read_excel('../input/Data.xlsx')
# viewing top 5 rows of data

data.head()
# checks data type of data

data.dtypes
#conversion of timestamp to date and time components

data['datetime'] = pd.to_datetime(data['Timestamp'], unit='ms')

data['Year'] = data.datetime.dt.year

data['Month'] = data.datetime.dt.month

data['Day'] = data.datetime.dt.day

data['Time'] = data.datetime.dt.time

data['Hour'] = data.datetime.dt.hour

data['Minutes'] = data.datetime.dt.minute

data['Seconds'] = data.datetime.dt.second

data['MicroSecond'] = data.datetime.dt.microsecond
data.head()
# Removal of unwanted colums

del data['datetime']

del data['Timestamp']

del data['Time']

del data['MicroSecond']
data.head()
# checking of stastical values of data

data.describe()
# ANalysis of Grid Status variable, as it takes only two values

data['Grid status'].value_counts()
fig, axes = plt.subplots(nrows=2,ncols=3)

fig.set_size_inches(12, 10)

sn.boxplot(data=data,x="Temperature",orient="v",ax=axes[0][0])

sn.boxplot(data=data,x="SOC",orient="v",ax=axes[0][1])

sn.boxplot(data=data,x="Grid status",ax=axes[0][2])

sn.boxplot(data=data,x="Equivalent cycle",orient="v",ax=axes[1][0])

sn.boxplot(data=data,x="SOH",orient="v",ax=axes[1][1])



axes[0][0].set(xlabel='Temperature',title="Box Plot of Temperature")

axes[0][1].set(xlabel='SOC', title="Box Plot SOC")

axes[0][2].set(xlabel='Grid status', title="Box Plot of Grid status")

axes[1][0].set(xlabel='Equivalent cycle', title="Box Plot Equivalent cycle ")

axes[1][1].set(xlabel='SOH', title="Box Plot SOH")

# Calculation of correlation coefficients

pair1 = np.corrcoef(data['Grid status'], data['SOC'])

print("Correlation between Grid status and  SOC")

pair1
# Relationship pf Grid Analysis and SOC

sn.boxplot(x='Grid status', y='SOC', data=data)

pair2 = np.corrcoef(data['Equivalent cycle'], data['SOH'])

print("Correlation between Equivalent cycle and  SOH")

pair2
pair3 = np.corrcoef(data['SOC'], data['Temperature'])

print("Correlation between SOC and  Temperature")

pair3
fig, axes = plt.subplots(nrows=2,ncols=2)

fig.set_size_inches(12, 10)

# prints the relationship between Equivalent cycle and SOH

sn.lineplot(data=data,x='Equivalent cycle',y= 'SOH',ax=axes[0][0])

#print the relation between SOC and temperature

sn.scatterplot(data=data, x='SOC',y= 'Temperature', ax=axes[0][1])

sn.lineplot(data=data, x='SOC',y= 'Temperature', ax=axes[1][0])

#print the relation between SOC and Grid Status

sn.lineplot(data=data, x='Grid status', y='SOC', ax=axes[1][1] )



axes[0][0].set(xlabel = 'Equivalent cycle', ylabel='SOH',title="Lineplot between Equivalent cycle and SOH")

axes[0][1].set(xlabel='SOC', ylabel='Temparature',title="scatterplot between SOC and Temperature")

axes[1][0].set(xlabel='SOC', ylabel='Temparature',title="Lineplot between SOC and Temperature")

axes[1][1].set(xlabel='Grid status', ylabel='SOC',title="Lineplot between Grid Status and SOC")
sn.scatterplot(data=data, x='Grid status', y='SOC')