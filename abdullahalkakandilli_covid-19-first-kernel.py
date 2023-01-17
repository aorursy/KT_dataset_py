# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 
import datetime as dt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
SeoulFloating = pd.read_csv("/kaggle/input/coronavirusdataset/SeoulFloating.csv")
case = pd.read_csv("/kaggle/input/coronavirusdataset/Case.csv")
SearchTrend = pd.read_csv("/kaggle/input/coronavirusdataset/SearchTrend.csv")
PatientRoute = pd.read_csv("/kaggle/input/coronavirusdataset/PatientRoute.csv")
TimeGender = pd.read_csv("/kaggle/input/coronavirusdataset/TimeGender.csv")
TimeAge = pd.read_csv("/kaggle/input/coronavirusdataset/TimeAge.csv")
Region = pd.read_csv("/kaggle/input/coronavirusdataset/Region.csv")
Weather = pd.read_csv("/kaggle/input/coronavirusdataset/Weather.csv")
Time = pd.read_csv("/kaggle/input/coronavirusdataset/Time.csv")
PatientInfo = pd.read_csv("/kaggle/input/coronavirusdataset/PatientInfo.csv")
TimeProvince = pd.read_csv("/kaggle/input/coronavirusdataset/TimeProvince.csv")
SearchTrend.tail(10)
SearchTrend.columns

SearchTrend.corr()
f,ax = plt.subplots(figsize=(5,5))
sns.heatmap(SearchTrend.corr(), annot=True, linewidths=20, fmt= ' .1f', ax=ax)
plt.show()
Time.head()
Time.columns
x_dateValues = Time[['date']]
#data frame

plt.plot(Time.date,Time.confirmed,zorder=1,color="black")
plt.plot(Time.date,Time.released,zorder=1,color="red")
plt.plot(Time.date,Time.test,zorder=1,color="blue")
plt.plot(Time.date,Time.negative,zorder=1,color="orange")
plt.show()
Time.test.plot(kind = 'line', color = 'g',label = 'test',linewidth=1,alpha = 0.9,grid = True,linestyle = ':')
Time.confirmed.plot(color = 'r',label = 'confirmed',linewidth=1, alpha = 0.9,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('time')              # label = name of label
plt.ylabel('test&confirmed count')
plt.title('Line Plot')            # title = title of plot
plt.show()
Time.plot(kind='scatter', x='confirmed', y='test',alpha = 0.5,color = 'red')
plt.xlabel('confirmed')              # label = name of label
plt.ylabel('test')
plt.title('confirmed test scatter plot')   
plt.show()
Time.confirmed.plot(kind = 'hist', bins = 50, figsize = (12,12), color = 'g')

# hist graph doens't mean anythin for this data.
plt.clf()
Time.head()
Time["test_date"] = ["after 16" if i > 16 else "before 16" for i in Time.time]
Time.loc[20:,["test_date", "confirmed"]]
Time.info()
#there is no nan data in this part
Time.describe()
#this doen't mean anything
Time.boxplot(column = 'confirmed')
data_time = Time.head()
data_time
melted = pd.melt(frame = data_time, id_vars = 'date', value_vars = ['test', 'confirmed'])
melted
melted.pivot(index = 'date', columns = 'variable', values = 'value')
data1 = Time.head()
data2 = Time.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row
data1 = Time['test'].head()
data2 = Time['confirmed'].head()
conc_data = pd.concat([data1,data2],axis =1)
conc_data
Time.dtypes

Time['date'] = Time['date'].astype('category')
Time.dtypes
Time.info()
#Time['test'].value_counts(dropna = False)
