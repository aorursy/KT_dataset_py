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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


speedviolation= pd.read_csv('../input/speed-camera-violations.csv')
speedviolation.head()
speedviolation.shape
speedviolation['ADDRESS'].value_counts()
speedviolation['CAMERA ID'].value_counts()
#Plotting a bar graph - No of Voilations / address

address = speedviolation['ADDRESS'].value_counts()
address = address[:10]
# print(address)
# print(address.values)
# print(address.index)
plt.figure(figsize=(10,8))
sns.barplot(x=address.index,y=address.values,alpha=.8)
plt.title(' Top 10 Speeidng Violations in Chicago at Different address')
plt.ylabel('Number of Violations', fontsize=10)
plt.xlabel('Address', fontsize=10)
plt.xticks(rotation=105)
plt.show()
# Plottng a bar graph showing the no of voilation / year.
# From the analysis we can conclude that 2015 has seen more violations in the last 5 years, but in general,the viloations have increased over a period of time.

speedviolation['YEAR'] = speedviolation['VIOLATION DATE'].apply(lambda date:pd.Period(date, freq='Q').year)
speedviolation.groupby('YEAR')['VIOLATIONS'].sum()
x_axis = (speedviolation.groupby('YEAR')['VIOLATIONS'].sum()).index
y_axis = (speedviolation.groupby('YEAR')['VIOLATIONS'].sum())
plt.figure(figsize=(10,5))
sns.barplot(x=x_axis,y=y_axis,alpha=0.8)
plt.title(' Number of Speeding Violations / Year')
plt.ylabel('Violatons', fontsize=10)
plt.xlabel('Year', fontsize=10)
plt.show()
# Plottng a bar graph showing the no of voilation / Quater.
# From the analysis we can conclude that 2015 has seen more violations in the last 5 years, but in general,the viloations have increased over a period of time.

speedviolation['QUATER'] = speedviolation['VIOLATION DATE'].apply(lambda date:pd.Period(date, freq='Q'))
speedviolation.groupby('QUATER')['VIOLATIONS'].sum()
x_axis = (speedviolation.groupby('QUATER')['VIOLATIONS'].sum()).index
y_axis = (speedviolation.groupby('QUATER')['VIOLATIONS'].sum())
plt.figure(figsize=(10,5))
sns.barplot(x=x_axis,y=y_axis,alpha=0.8)
plt.title(' Number of Speeding Violations / Quater')
plt.ylabel('Violatons', fontsize=10)
plt.xlabel('Quater', fontsize=10)
plt.xticks(rotation=105)
plt.show()
#Top 10 least viloations / day
speedviolation.groupby('VIOLATION DATE')['VIOLATIONS'].sum().sort_values(ascending = True)[:10]
leastviolationdates=speedviolation.groupby('VIOLATION DATE')['VIOLATIONS'].sum().sort_values(ascending = True)[:10]
x_axis = leastviolationdates.index
y_axis = leastviolationdates
plt.figure(figsize=(10,5))
sns.barplot(x=x_axis,y=y_axis,alpha=0.4)
plt.title(' Least Number of Speed violations in a day - In last 5 years ')
plt.xlabel('Date', fontsize=10)
plt.ylabel('No of Violatons', fontsize=10)
plt.xticks(rotation=105)
plt.show()
#Average no of speed violations 
speedviolation.groupby('VIOLATION DATE')['VIOLATIONS'].mean()
#Max no of speed violations
speedviolation.groupby('VIOLATION DATE')['VIOLATIONS'].max()
#Min no of speed violations
speedviolation.groupby('VIOLATION DATE')['VIOLATIONS'].min()
#No of speed viloations on Christmas day of each year( as it is one of most travelled day)

speedviolation['DTE']=speedviolation['VIOLATION DATE'].apply(lambda date:pd.Period(date,freq='D'))
ChristmasSpeedViolations= speedviolation.groupby('DTE')['VIOLATIONS'].sum().filter(like='12-25')
#ChristmasSpeedViolations.filter(like='12-25')
x_axis = ChristmasSpeedViolations.index
y_axis = ChristmasSpeedViolations
plt.figure(figsize=(10,5))
sns.barplot(x=x_axis,y=y_axis,alpha=0.8)
plt.title(' Number of Speeding Violations on Christmas day of each year ')
plt.ylabel('No of Violatons', fontsize=10)
plt.xlabel('DTE', fontsize=10)
plt.show()
#No of speed viloations on New Years day of each year( as it is one of most travelled day)
speedviolation['DTE']=speedviolation['VIOLATION DATE'].apply(lambda date:pd.Period(date,freq='D'))
NewYearSpeedViolations= speedviolation.groupby('DTE')['VIOLATIONS'].sum().filter(like='01-01')
x_axis = NewYearSpeedViolations.index
y_axis = NewYearSpeedViolations
plt.figure(figsize=(10,5))
sns.barplot(x=x_axis,y=y_axis,alpha=0.8)
plt.title(' Number of Speeding Violations on Newyear day of each year ')
plt.ylabel('No of Violatons', fontsize=10)
plt.xlabel('DTE', fontsize=10)
plt.show()

