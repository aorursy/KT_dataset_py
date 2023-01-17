%reset -f

import pandas as pd

import numpy as np

import os
import pandas as pd
df=pd.read_csv('../input/border-crossing-entry-data/Border_Crossing_Entry_Data.csv') 
df.head()
df.dtypes
New_Columns={

             'Port Name' : 'port_name',

             'State'     : 'state',

             'Port Code' : 'port_code',

             'Border'    : 'border',

             'Date'      : 'date',

             'Measure'   : 'measure',

             'Value'     : 'value',

             'Location'  : 'location'

             }

df.rename(New_Columns,inplace='True',axis=1)
df.columns.values
df['date']=pd.to_datetime(df.date)
df.dtypes
df['border'].nunique()
df['port_code'].nunique()
df.sort_values('port_code', ascending ='False')
df.head(10)
df['hour']=df['date'].dt.hour
#  Version 0.18.1 and newer

# There is now a new convenience method dt.weekday_name to do the above

# df['dayName']=df['date'].dt.weekday_name
df['weekday'] = df['date'].dt.dayofweek
df['month']=df['date'].dt.month
df['year']=df['date'].dt.year
df['quarter']=df['date'].dt.quarter
df['year'].unique()
df['quarter'].unique()
df['weekday'] = df ['weekday'].map ({ 

                      0: 'Monday',

                      1: 'Tuesday',

                      2: 'Wednesday',

                      3: 'Thursday',

                      4: 'Friday',

                      5: 'Saturday',

                      6: 'Sunday'

                    }

                  )
df['weekday'].unique()
df.head(5)
df.columns.values
df['date']  # checking if it has timestamp
df.isnull().any()  # check if we have any null values
borders=df['border'].unique()

print(borders)
df.port_name.nunique()
df.port_code.nunique()
ports=df[['port_name','port_code','location']].nunique()  

ports
df.location.nunique()
df['measure'].unique()
Passenger=['Personal Vehicle Passengers','Personal Vehicles','Pedestrians','Train Passengers','Bus Passengers']
df.loc[df['measure'].isin(Passenger),'Type'] ='Passenger'
df.loc[~df['measure'].isin(Passenger),'Type'] ='Vehicles'
df['Type'].unique()
# plotting data, follwoing things required

from sklearn.preprocessing import StandardScaler

# 1.3.2 Split dataset

from sklearn.model_selection import train_test_split

# 1.3.3 Class to develop kmeans model

from sklearn.cluster import KMeans
# 1.2 For plotting

import matplotlib.pyplot as plt        # ploty web based graph -grpahics

import matplotlib

import matplotlib as mpl     # For creating colormaps

import seaborn as sns  # for building graphics -windows based package...its easier than plotli



plt.figure(figsize=(15,3))  #  to increase graph length
#1) What are total passengers travelled accross bonders



sns.barplot(data=df[df['Type']=='Passenger'],x='border',y='value', estimator =sum)
#2) What are total vehicles travelled accross bonders



sns.barplot(data=df[df['Type']=='Vehicles'], x='border',y='value',estimator = sum)
# 3) Can you get US Canada first and then US Mexico later 

sns.barplot(data=df[df['Type']=='Vehicles'], x='border',y='value', order= ['US-Canada Border','US-Mexico Border'],estimator = sum)

#4) What are diffrrent type of meaures and their sum for both borders- plot bar chart only for vehlcle



import matplotlib.pyplot as plt

plt.figure(figsize=(15,3))



sns.barplot(data=df[df['Type']=='Vehicles'], x='measure',y='value',hue='border',estimator =sum)
#5)What are diffrrent type of meaures and their sum for both borders- plot bar chart only for passengers



plt.figure(figsize=(15,3))

sns.barplot(data=df[df['Type']=='Passenger'], x='measure',y='value', hue='border',estimator =sum)
#6) find sum of values for all measures



df.groupby('measure').sum().value
#7) Use group by method to find sum of values for all passengers for borders

data1= df[df.Type=='Passenger'].groupby(['border','measure']).sum().value.reset_index()



data1
plt.figure(figsize=(15,3))

sns.barplot(data=data1, x='measure',y='value',hue='border')

#8) which state in US has max number and lowest number of passengers corssed 



stateData=df.groupby(['state','Type'])['value'].sum().sort_values(ascending=False).reset_index()



#data111= df.groupby(['border','measure']).sum().value.reset_index()

stateData.head()
import matplotlib.pyplot as plt



plt.figure(figsize=(15,3))

sns.barplot(data=stateData[stateData.Type=='Passenger'],x='state',y='value')
#8) which state has highest and lowest numebr vehicles travelled



StateVehicle=df.groupby(['state','Type'])['value'].sum().sort_values(ascending=False).reset_index()

StateVehicle.head()
import matplotlib.pyplot as plt

plt.figure(figsize=(15,3))



sns.barplot(data=StateVehicle[StateVehicle.Type=='Vehicles'],y = 'state', x= 'value')
#9) find and plot -which port has highest and lowest number vehicles and passengers travelled



portdata=df.groupby(['port_name','Type'])['value'].sum().sort_values(ascending=False).reset_index()

portdata.head()
import matplotlib.pyplot as plt

plt.figure(figsize=(15,40))

sns.barplot(data=portdata[portdata.Type=='Passenger'],y='port_name',x='value')
#10) which day of week has highest entries of passengers and vehicle



day_data=df.groupby(['weekday','Type'])['value'].sum().sort_values(ascending=False).reset_index()

day_data.head()
import matplotlib.pyplot as plt

plt.figure(figsize=(15,6))



#sns.barplot(data=day_data[day_data.Type=='Passenger'],x='weekday',y='value')

sns.barplot(data=day_data,x='weekday',y='value',hue='Type')
#11) which month has highest entries of passengers and vehicle



month_data=df.groupby(['month','border','Type'])['value'].sum().sort_values(ascending=False).reset_index()

month_data.head()
import matplotlib.pyplot as plt

plt.figure(figsize=(15,3))

sns.barplot(data=month_data,x='month',y='value',hue='border',estimator=sum)
#12) which year has highest entries of passengers and vehicle across borders

year_data=df.groupby(['year','Type'])['value'].sum().sort_values(ascending=False).reset_index()

year_data.head()
import matplotlib.pyplot as plt

plt.figure(figsize=(15,3))

#sns.barplot(data=year_data,x='year',y='value',hue='Type')

sns.barplot(data=year_data[year_data.Type=='Passenger'],x='year',y='value')
import matplotlib.pyplot as plt

plt.figure(figsize=(15,3))

#sns.barplot(data=year_data,x='year',y='value',hue='Type')

sns.barplot(data=year_data[year_data.Type=='Vehicles'],x='year',y='value')
group1=df.groupby(['month','year','Type'])['value'].sum().reset_index()

group1.head()
#mask = year_gruped['Type'].isin ['Passenger']



year_group= group1[group1.Type=='Passenger']

year_group.head()

num_columns = year_group.select_dtypes(include = ['float64', 'int64']).copy()

num_columns.head()

num_monthData=num_columns.groupby(['month','year'])['value'].sum().sort_values(ascending=False).reset_index()

num_monthData.head()
sns.set(style="whitegrid")

#sns.barplot(data=year_data[year_data.Type=='Vehicles'],x='year',y='value')

plt.figure(figsize=(15,10))

sns.lineplot(data=num_monthData,x='month',y='value',hue='year',palette="tab10")
 
import pandas as pd

Border_Crossing_Entry_Data = pd.read_csv("../input/border-crossing-entry-data/Border_Crossing_Entry_Data.csv")