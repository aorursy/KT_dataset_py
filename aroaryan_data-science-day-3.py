# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from datetime import date, timedelta
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_pgen1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df_pgen2 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
df_pgen3 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')
df_pgen4 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
df_pgen1['DATE_TIME'] = pd.to_datetime(df_pgen1['DATE_TIME'], format = '%d-%m-%Y %H:%M')
df_pgen1['DATE'] = df_pgen1['DATE_TIME'].apply(lambda x:x.date())
df_pgen1['TIME'] = df_pgen1['DATE_TIME'].apply(lambda x:x.time())
df_pgen1['DATE'] = pd.to_datetime(df_pgen1['DATE'],format = '%Y-%m-%d')
df_pgen1['HOUR'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.hour
df_pgen1['MINUTES'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.minute
df_pgen2['DATE_TIME'] = pd.to_datetime(df_pgen2['DATE_TIME'], format = '%Y-%m-%d %H:%M:%S')
df_pgen2['DATE'] = df_pgen2['DATE_TIME'].apply(lambda x:x.date())
df_pgen2['TIME'] = df_pgen2['DATE_TIME'].apply(lambda x:x.time())
df_pgen2['DATE'] = pd.to_datetime(df_pgen2['DATE'],format = '%Y-%m-%d')
df_pgen2['HOUR'] = pd.to_datetime(df_pgen2['TIME'],format='%H:%M:%S').dt.hour
df_pgen2['MINUTES'] = pd.to_datetime(df_pgen2['TIME'],format='%H:%M:%S').dt.minute
df_pgen3['DATE_TIME'] = pd.to_datetime(df_pgen3['DATE_TIME'], format = '%Y-%m-%d %H:%M:%S')
df_pgen3['DATE'] = df_pgen3['DATE_TIME'].apply(lambda x:x.date())
df_pgen3['TIME'] = df_pgen3['DATE_TIME'].apply(lambda x:x.time())
df_pgen3['DATE'] = pd.to_datetime(df_pgen3['DATE'],format = '%Y-%m-%d')
df_pgen3['HOUR'] = pd.to_datetime(df_pgen3['TIME'],format='%H:%M:%S').dt.hour
df_pgen3['MINUTES'] = pd.to_datetime(df_pgen3['TIME'],format='%H:%M:%S').dt.minute
df_pgen4['DATE_TIME'] = pd.to_datetime(df_pgen4['DATE_TIME'], format = '%Y-%m-%d %H:%M:%S')
df_pgen4['DATE'] = df_pgen4['DATE_TIME'].apply(lambda x:x.date())
df_pgen4['TIME'] = df_pgen4['DATE_TIME'].apply(lambda x:x.time())
df_pgen4['DATE'] = pd.to_datetime(df_pgen4['DATE'],format = '%Y-%m-%d')
df_pgen4['HOUR'] = pd.to_datetime(df_pgen4['TIME'],format='%H:%M:%S').dt.hour
df_pgen4['MINUTES'] = pd.to_datetime(df_pgen4['TIME'],format='%H:%M:%S').dt.minute
print(len(df_pgen1['SOURCE_KEY'].unique()))
print(len(df_pgen3['SOURCE_KEY'].unique()))
#Exploring Data 
df_pgen1['DC_POWER'].mean()
df_pgen1[df_pgen1['SOURCE_KEY'] == 'wCURE6d3bPkepu2']['DC_POWER'].mean()
df_pgen1.head()
df_pgen1.tail()
df_pgen1.value_counts()
df_pgen1['DATE_TIME'].value_counts()
df_pgen1.describe()



#Mean Task1 Subtask 1

print("Mean of Daily Yield of Plant 1:", df_pgen1['DAILY_YIELD'].mean())
print("Mean of Daily Yield of Plant 2:", df_pgen3['DAILY_YIELD'].mean())
tsum = df_pgen1['DAILY_YIELD'].sum()+df_pgen3['DAILY_YIELD'].sum()
#type(tsum)
tlen = len(df_pgen3['SOURCE_KEY'])+len(df_pgen1['SOURCE_KEY'])
#type(tlen)
tmean = float(tsum)/float(tlen)
print("Mean of Daily Yield of Plant 1 and Plant 2:", tmean )

#Irradiation Task 1 Subatask 2
print("Plant 1:")
start_date = date(2020, 5, 15)
end_date = date(2020, 6, 17)
delta = timedelta(days=1)
print("Date       :  Total Irradiation")
df_pgen2['IRRADIATION'][df_pgen2['DATE']=='start_date'].sum()
while start_date <= end_date:
    print (str(start_date),":",df_pgen2['IRRADIATION'][df_pgen2['DATE']==str(start_date)].sum())
    start_date += delta
#Irradiation Task 1 Subatask 2
print("Plant 2:")
start_date = date(2020, 5, 15)
end_date = date(2020, 6, 17)
delta = timedelta(days=1)
print("Date       :  Total Irradiation")
while start_date <= end_date:
    print (str(start_date),":",df_pgen4['IRRADIATION'][df_pgen4['DATE']==str(start_date)].sum())
    start_date += delta
#Max Task 1 Subtask 3
#df_pgen2.info()
print("Plant 1: Maximum Ambient Temp :", df_pgen2['AMBIENT_TEMPERATURE'].max())
print("Plant 2: Maximum Ambient Temp :", df_pgen4['AMBIENT_TEMPERATURE'].max())
print("Plant 1: Maximum Module Tempe :", df_pgen2['MODULE_TEMPERATURE'].max())
print("Plant 2: Maximum Module Tempe :", df_pgen4['MODULE_TEMPERATURE'].max())
# inverters Task 1 Subtask 4
print("Plant 1:",len(df_pgen1['SOURCE_KEY'].unique()))
print("Plant 2:",len(df_pgen3['SOURCE_KEY'].unique()))
#MinMax Task1 Subtask5
print("PLANT 1:")
start_date = date(2020, 5, 15)
end_date = date(2020, 6, 17)
delta = timedelta(days=1)
print("Date       :  Max")
while start_date <= end_date:
    print (str(start_date),":",df_pgen1['DC_POWER'][df_pgen1['DATE']==str(start_date)].max())
    start_date += delta    
start_date = date(2020, 5, 15)
end_date = date(2020, 6, 17)
delta = timedelta(days=1)
print()
print()
print("Date       :  Min")
while start_date <= end_date:
    print (str(start_date),":",df_pgen1['DC_POWER'][df_pgen1['DATE']==str(start_date)].min())
    start_date += delta
    
#MinMax Task1 Subtask5
print("PLANT 2:")
start_date = date(2020, 5, 15)
end_date = date(2020, 6, 17)
delta = timedelta(days=1)
print("Date       :  Max")
#df_pgen1['IRRADIATION'][df_pgen1['DATE']=='start_date'].sum()

while start_date <= end_date:
    print (str(start_date),":",df_pgen3['DC_POWER'][df_pgen3['DATE']==str(start_date)].max())
    start_date += delta
    
start_date = date(2020, 5, 15)
end_date = date(2020, 6, 17)
delta = timedelta(days=1)
print()
print()
print("Date       :  Min")
#df_pgen1['IRRADIATION'][df_pgen1['DATE']=='start_date'].sum()

while start_date <= end_date:
    print (str(start_date),":",df_pgen3['DC_POWER'][df_pgen3['DATE']==str(start_date)].min())
    start_date += delta
    
# power Task 1 Subtask 6
#print(df_pgen1['AC_POWER'].max())

df_pgen1['AC_POWER'].argmax() 
print("Plant 1:")
print("Maximum DC Invertor :", df_pgen1['SOURCE_KEY'].values[df_pgen1['DC_POWER'].argmax()])
print("Minimum DC Invertor :", df_pgen1['SOURCE_KEY'].values[df_pgen1['DC_POWER'].argmin()])
print("Maximum AC Generator:", df_pgen1['SOURCE_KEY'].values[df_pgen1['AC_POWER'].argmax()])
print("Minimum AC Generator:", df_pgen1['SOURCE_KEY'].values[df_pgen1['AC_POWER'].argmin()])

print("Plant 2:")
print("Maximum DC Invertor :", df_pgen3['SOURCE_KEY'].values[df_pgen3['DC_POWER'].argmax()])
print("Minimum DC Invertor :", df_pgen3['SOURCE_KEY'].values[df_pgen3['DC_POWER'].argmin()])
print("Maximum AC Generator:", df_pgen3['SOURCE_KEY'].values[df_pgen3['AC_POWER'].argmax()])
print("Minimum AC Generator:", df_pgen3['SOURCE_KEY'].values[df_pgen3['AC_POWER'].argmin()])

print("Overall:")
if(df_pgen3['DC_POWER'].argmax()>df_pgen1['DC_POWER'].argmax()):
    print("Maximum DC/AC Invertor :", df_pgen3['SOURCE_KEY'].values[df_pgen3['DC_POWER'].argmax()])
else:
    print("Maximum DC/AC Invertor :", df_pgen1['SOURCE_KEY'].values[df_pgen1['DC_POWER'].argmax()])

if(df_pgen3['DC_POWER'].argmin()<df_pgen1['DC_POWER'].argmin()):
    print("Minimum DC/AC Invertor :", df_pgen3['SOURCE_KEY'].values[df_pgen3['DC_POWER'].argmin()])
else:
    print("Minimum DC/AC Invertor :", df_pgen1['SOURCE_KEY'].values[df_pgen1['DC_POWER'].argmin()])


#sort Task 1 Subtask 7
#First Plant
print(df_pgen1.sort_values(by='DC_POWER', ascending=False))


#Second Plant
df_pgen3.sort_values(by='DC_POWER', ascending=False)
# Null Task1 Subtask 8
df_pgen1.isnull().sum()
df_pgen2.isnull().sum()
df_pgen3.isnull().sum()
df_pgen4.isnull().sum()

#df_pgen1['DATE'].value_counts().sort_index()
#there are missing values as there are less number of time intervals recorded than there should be
import matplotlib.pyplot as plt
_, ax = plt.subplots(1, 1, figsize=(18, 9))
df_subset = df_pgen1[df_pgen1['DATE']=='2020-05-21']


ax.plot(df_subset.DATE_TIME,
        df_subset.DC_POWER/10,
        marker = 'o',
        linestyle='',
        label='DC_POWER'
       )

ax.plot(df_subset.DATE_TIME,
        df_subset.AC_POWER,
        marker = 'o',
        linestyle='',
        label='AC_POWER'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC Power and AC Power over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('Power')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))
df_subset = df_pgen1[df_pgen1['DATE']=='2020-05-21']


ax.plot(df_subset.DATE_TIME,
        df_subset.DC_POWER/10,
        marker = '.',
        linestyle='',
        label='DC_POWER'
       )

ax.plot(df_subset.DATE_TIME,
        df_subset.AC_POWER,
        marker = '.',
        linestyle='',
        label='AC_POWER'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC Power and AC Power over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('Power')
plt.show()
import matplotlib.pyplot as plt
df_subset = df_pgen1[['SOURCE_KEY','AC_POWER']]
df_pgen1.plot(x ='SOURCE_KEY', y='AC_POWER', kind = 'bar')

'''
df_subset.plot(df_subset.SOURCE_KEY,
        df_subset.AC_POWER,
        marker = 'o',
        linestyle='',
        label='AC_POWER'
       )


ax.plot(df_subset.DATE_TIME,
        df_subset.AC_POWER,
        marker = '.',
        linestyle='',
        label='AC_POWER'
       )
       

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('DC Power and AC Power over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('Power')
plt.show()
#df.plot(x ='Unemployment_Rate', y='Stock_Index_Price', kind = 'scatter')
'''

#df_subset