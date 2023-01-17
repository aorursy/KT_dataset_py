# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.plotly as py
import plotly
import os
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Question 10
data_sch=pd.read_csv("../input/data-science-for-good/2016 School Explorer.csv")
#data cleaning 
data_sch.shape
data_sch[data_sch.iloc[:,:] == 'N/A'] = np.nan
data_s=data_sch.iloc[:,3:] #subsetting data to remove first 3 columns 
#data_sch[data_sch.isna()]
data_s=data_s.dropna()   # drop the columns with na values 
data_s=data_s.reset_index(drop=True) # reset the index 
#list(data_s)
# formatting data
a=list(data_s.columns[15:24])
for i in a:
    data_s[i] = data_s[i].str.rstrip('%').astype('float') / 100.0  # converting the percentage to float values 
data_s['School Income Estimate'] = data_s['School Income Estimate'].str.rstrip('$').replace(',','')
data_s['School Income Estimate'] = data_s['School Income Estimate'].str.replace('$','')
data_s['School Income Estimate'] = data_s['School Income Estimate'].str.replace(',','').astype('float')  #converting money table to float 

#data analysis 
data1=data_s.groupby(['Community School?'])['Percent ELL',
 'Percent Asian',
 'Percent Black',
 'Percent Hispanic',
 'Percent Black / Hispanic',
 'Percent White'].mean()

data2=data_s.groupby(['District'])['Economic Need Index'].mean()
data2=data2.to_frame()
data2.columns.values[0]='Economic Need Index'
data3=data_s.groupby(['District'])[
 'Percent Asian',
 'Percent Black',
 'Percent Hispanic',
 'Percent White'].mean()

data4=data_s.groupby(['District'])['School Income Estimate'].mean()
data4=data4.to_frame()
data4.columns.values[0]='School Income Estimate'

data5=data_s.groupby(['City'])['Percent Asian'].mean()
data5=data5.to_frame()
data5.columns.values[0]='Average Percentage of Asians'

data6=data_s.groupby(['Supportive Environment Rating'])['Percent of Students Chronically Absent'].mean()
data6=data6.to_frame()
data6.columns.values[0]='Percent of Students Chronically Absent'

#Visualisation and questions 
plt.figure(1)
data1.plot(kind='bar',figsize=(10,10))

plt.figure(2)
data2.plot(kind='bar',title='How does  economic index  vary in the schools districtwise ?')

plt.figure(3)
data3.plot(kind='bar',figsize=(20,10),title='How does the distribution of students vary districtwise?')

plt.figure(5)
data4.plot(kind='bar',figsize=(20,10),title='How does  School Income Estimate  vary in the schools districtwise ?')

plt.figure(7)
data5.plot(kind='bar',figsize=(20,10),title='How does  percentage of asians   vary in the schools citywise ?')

plt.figure(9)
data6.plot(kind='bar',figsize=(20,10),title='How does  percentage of chronically absent students  vary with supportive environment ?')

#Question 11
df1=pd.read_csv("../input/montcoalert/911.csv")
df1['Category'] = df1['title'].apply(lambda x: x.split(':')[0])
df1['timeStamp'] = pd.to_datetime(df1['timeStamp'])
df1['day'] = df1['timeStamp'].apply(lambda x: x.dayofweek)
df1['month'] = df1['timeStamp'].apply(lambda x: x.month)
df1['h'] = pd.DataFrame([(x.hour) for x in df1['timeStamp']])
df1['Part of day'] = pd.cut(df1['h'], bins=[0, 6, 12, 18, 24], include_lowest=True, labels=['0-6 hrs', '6-12 hrs', '12-18 hrs', '18-24'])
mymap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df1['day']= df1['day'].apply(lambda s: mymap.get(s) if s in mymap else s)
df1.head()
plt.figure()
sb.countplot(df1['Category']).set_title('Number of calls recieved in each category')
plt.figure()
sb.countplot(df1['day']).set_title('Number of calls received by Day of Week')
plt.figure()
sb.countplot(df1['month']).set_title('Number of calls received by Month')
plt.figure()
sb.countplot(df1['twp'], order=pd.value_counts(df1['twp']).iloc[:10].index).set_title('Number of calls by towns (top 10)')
plt.figure()
sb.countplot(df1['Part of day']).set_title('Number of calls received by Part of Day')

