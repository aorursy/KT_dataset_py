# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/AirQualityUCI_req.csv')
data.index = pd.DatetimeIndex(data.Date, dayfirst=True).strftime('%Y-%m-%d')
data = data.drop(['Date' ], 1)
cols = data.columns
data = data[data[cols] > 0]
data = data.fillna(method='ffill')
data.head()
daily_data = data.groupby(data.index).mean()
daily_data.head()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sb
sb.set(style="darkgrid")
sb.clustermap(daily_data)
cols = ['NO2(GT)', 'C6H6(GT)', 'PT08.S4(NO2)', 'PT08.S3(NOx)', 'PT08.S5(O3)']
sb.clustermap(daily_data[cols])
g = sb.jointplot("C6H6(GT)", "NO2(GT)", data = daily_data, kind="reg")
g = sb.jointplot("PT08.S4(NO2)", "NO2(GT)", data = daily_data, kind="reg")
g = sb.jointplot("PT08.S3(NOx)", "NO2(GT)", data = daily_data, kind="reg")
g = sb.jointplot("PT08.S5(O3)", "NO2(GT)", data = daily_data, kind="reg")
cols = ['NO2(GT)', 'C6H6(GT)', 'PT08.S4(NO2)']
sb.pairplot(daily_data[cols])
set1 = ['NO2(GT)']
set2 = ['C6H6(GT)' ]
set3 = ['PT08.S4(NO2)']
sb.lineplot(data=daily_data[set1], linewidth=2.5)
sb.lineplot(data=daily_data[set2], linewidth=2.5)
sb.lineplot(data=daily_data[set3], linewidth=2.5)
import datetime
daily_data['Date'] = daily_data.index
daily_data['Day'] = daily_data['Date'].map( lambda x : datetime.datetime.strptime(x, '%Y-%m-%d').strftime('%A'))
daily_data['Month'] = daily_data['Date'].map( lambda x : datetime.datetime.strptime(x, '%Y-%m-%d').strftime('%m'))
daily_data.head()
cats = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_data = daily_data.groupby('Day').mean()
weekly_data['Day'] = weekly_data.index
weekly_data = weekly_data.reindex(cats)
weekly_data
sb.barplot(x="Day", y=set1[0], data=weekly_data)
sb.barplot(x="Day", y=set2[0], data=weekly_data)
sb.barplot(x="Day", y=set3[0], data=weekly_data)
monthly_data = daily_data.groupby('Month').mean()
monthly_data['Month'] = monthly_data.index
monthly_data
sb.barplot(x="Month", y=set1[0], data=monthly_data)
sb.barplot(x="Month", y=set2[0], data=monthly_data)
sb.barplot(x="Month", y=set3[0], data=monthly_data)
d = weekly_data[set1[0]].values
plt.plot(d)
plt.ylabel(set1[0])
plt.xticks([i for i in range(len(d))], cats, rotation=20)
d = weekly_data[set2[0]].values
plt.plot(d)
plt.ylabel(set2[0])
plt.xticks([i for i in range(len(d))], cats, rotation=20)
d = weekly_data[set3[0]].values
plt.plot(d)
plt.ylabel(set3[0])
plt.xticks([i for i in range(len(d))], cats, rotation=20)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
d = monthly_data[set1[0]].values
plt.plot(d)
plt.ylabel(set1[0])
plt.xticks([i for i in range(len(d))], months, rotation=20)
d = monthly_data[set2[0]].values
plt.plot(d)
plt.ylabel(set2[0])
plt.xticks([i for i in range(len(d))], months, rotation=20)
d = monthly_data[set3[0]].values
plt.plot(d)
plt.ylabel(set3[0])
plt.xticks([i for i in range(len(d))], months, rotation=20)