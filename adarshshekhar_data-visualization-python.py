# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
crimeData=pd.read_csv('../input/train.csv')
print(crimeData.shape)
crimeData.head()
crimeData['Dates']=pd.to_datetime(crimeData["Dates"])
crimeData['Year']=crimeData['Dates'].dt.year
crimeData['Month']=crimeData['Dates'].dt.month
crimeData['Hour']=crimeData['Dates'].dt.hour
month_map={1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
crimeData['Month'].replace(month_map, inplace=True)
crimeData['Year']=crimeData['Year'].astype(str)
crimeData['Hour']=crimeData['Hour'].astype(str)
crimeData.head()
YearlyData = pd.DataFrame(crimeData["Year"].value_counts())
MonthlyData = pd.DataFrame(crimeData["Month"].value_counts())
WeeklyData = pd.DataFrame(crimeData["DayOfWeek"].value_counts(sort=False))
HourlyData = pd.DataFrame(crimeData["Hour"].value_counts())
DistrictData = pd.DataFrame(crimeData["PdDistrict"].value_counts())
plt.figure(figsize=(16,10))
ax1 =  plt.subplot2grid((2,2),(0,0))
ax1.set_title('Weekly')
sns.barplot(x=WeeklyData.index, y="DayOfWeek", data=WeeklyData, order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
ax2 =  plt.subplot2grid((2,2),(0,1))
ax2.set_title('Hourly')
sns.barplot(x=HourlyData.index, y="Hour", data=HourlyData, order=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23'])
ax3 =  plt.subplot2grid((2,2),(1,0))
ax3.set_title('Monthly')
sns.barplot(x=MonthlyData.index, y="Month", data=MonthlyData)
ax4 =  plt.subplot2grid((2,2),(1,1))
ax4.set_title('Yearly')
sns.barplot(x=YearlyData.index, y="Year", data=YearlyData)
plt.figure(figsize=(16,10))
ax1 =  plt.subplot2grid((1,1),(0,0))
ax1.set_title('Police District')
sns.barplot(x=DistrictData.index, y="PdDistrict", data=DistrictData)
tmp=pd.DataFrame(crimeData.groupby(['Year','Month','DayOfWeek','Hour','PdDistrict']).size(), columns=['count'])
tmp.reset_index(inplace=True)
tmp.head()
df_cr=pd.DataFrame(crimeData['Category'].value_counts())
df_cr.tail()
plt.figure(figsize=(16,10))
ax1 =  plt.subplot2grid((1,2),(0,0))
ax1.set_title('Top 10', size=16)
sns.barplot(x=df_cr.head(10).index, y='Category', data=df_cr.head(10))
ax1.set_xticklabels(ax1.xaxis.get_ticklabels(), rotation=90)
ax2 =  plt.subplot2grid((1,2),(0,1))
ax2.set_title('Bottom 10', size=16)
sns.barplot(x=df_cr.tail(10).index, y='Category', data=df_cr.tail(10))
ax2.set_xticklabels(ax2.xaxis.get_ticklabels(), rotation=90)
top10cc=pd.Series(df_cr.head(10).index)
top10cc
top10=crimeData[crimeData['Category'].isin(top10cc)]
top10.describe(include='all')
tmp=pd.DataFrame(top10.groupby(['PdDistrict','Category']).size(), columns=['count'])
tmp.reset_index(inplace=True)
tmp=tmp.pivot(index='PdDistrict',columns='Category',values='count')
fig, axes = plt.subplots(1,1,figsize=(15,15))
tmp.plot(ax=axes,kind='bar', stacked=True)
