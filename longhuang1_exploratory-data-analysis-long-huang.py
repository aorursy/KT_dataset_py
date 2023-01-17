import pandas as pd

import numpy as np

import collections as c

import seaborn as sns

import matplotlib.pyplot as plt

import datetime

data=pd.read_csv('../input/ridershippertrip-pertripaugustweekdays99/Ridershippertrip_PerTrip-AugustWeekdays9-9.csv')

data.drop(columns='No',inplace=True)

data
numeric_features = data.select_dtypes(include=[np.number])

numeric_features.columns

a=numeric_features.describe() 
count=[]

for col in numeric_features.columns:

    count.append(len(data[col].unique()))

count=pd.DataFrame(count)

count.columns=['unique']

b=count.T

b.columns=a.columns

a=a.append(b) 

a
count=[]

for col in data.columns:

    count.append(len(data[col].unique()))

count=pd.DataFrame(count)

count=count.T

count.columns=data.columns
day=data[data['Date']=='8/24/2020']

day_des=day.describe()



day_count=[]

for col in day.columns:

    day_count.append(len(day[col].unique()))

day_count=pd.DataFrame(day_count)

day_count=day_count.T

day_count.columns=day.columns

day_des=day_des.append(day_count)
#Heapmap

cols=['P-Stops','M-Stops','Total in','Total out','Load avg','Max','PM'] 

df11=data[cols]

corr=df11.corr()

fig, ax = plt.subplots(figsize = (9,6))

sns.heatmap(corr,annot=True)
sns.pairplot(df11, kind="reg")
def retime(day):

    time=day['Sched. start'].str.split(':',expand=True) 

    day=day.join(time)

    day.drop(columns=[2],inplace=True)

    day.rename(columns={0:'hour',1:'mins'},inplace=True)

    

    day['Date']=pd.to_datetime(day['Date'])

    day['hour']=day['hour'].astype('int')

    offset1 = datetime.timedelta(days=1)

    offset2 = datetime.timedelta(days=2)

    

    day.loc[day['hour']==24,'Date']+=offset1

    day.loc[day['hour']==25,'Date']+=offset2

    

    day['Date']=day['Date'].astype('str')

    day['hour']=day['hour'].astype('str')

    

    day['hour']=day['hour'].replace('24','1')

    day['hour']=day['hour'].replace('25','2')

    day['mins']=day['mins'].astype('str')

    day['time']=day['Date'].str[0:10]+' '+day['hour']+':'+day['mins']+':'+'00'

    

    day['time']=pd.to_datetime(day['time'])

    

    return day



day=data[data['Date']=='8/24/2020'] #Select 8/24/2020

day=retime(day)
plt.scatter(day['hour'], day['PM'])

plt.ylabel('PM')

plt.xlabel('hour')
#Timing scatter diagram 

plt.figure(figsize=(12,9),dpi=300)

plt.subplot(3,1,1)

plt.scatter(day['hour'], day['PM'])

plt.xticks (np.linspace(0,23,24))

plt.ylabel('PM')



plt.subplot(3,1,2)

plt.scatter(day['hour'], day['Total in'])

plt.xticks (np.linspace(0,23,24))

plt.ylabel('Total in')



plt.subplot(3,1,3)

plt.scatter(day['hour'], day['Total out'])

plt.xticks (np.linspace(0,23,24))

plt.ylabel('Total out')

plt.xlabel('hour')
df=retime(data)



col='hour' #Block'#'Line'

sum_=df.groupby(df[col])['P-Stops','M-Stops','PM','Total in','Total out','Load avg'].sum()

st=sum_.reset_index()

st.hour=st.hour.astype('int')

st.sort_values(by='hour',inplace=True)


col='Block'

sum_=df.groupby(df[col])['P-Stops','M-Stops','PM','Total in','Total out','Load avg'].sum()

sum_
col='Line' 

sum_=df.groupby(df[col])['P-Stops','M-Stops','PM','Total in','Total out','Load avg'].sum()

sum_
cols=['P-Stops','M-Stops','PM','Total in','Total out','Load avg']

for col in cols:

    plt.plot(st['hour'],st[col],label=col) # Polyline1  

plt.xticks (np.linspace(0,23,24))

plt.xlabel(u'hour')

plt.ylabel(u'sum')

plt.legend() #Legend, loc is the location
#daily Passenger Flow

df1=df[(df['time']>='2020-08-03 00:00:00')& (df['time']<='2020-08-08 00:00:00')]

df2=df[(df['time']>='2020-08-10 00:00:00')& (df['time']<='2020-08-15 00:00:00')]

df3=df[(df['time']>='2020-08-17 00:00:00')& (df['time']<='2020-08-22 00:00:00')]

df4=df[(df['time']>='2020-08-24 00:00:00')& (df['time']<='2020-09-01 00:00:00')] 
def groupd(df): #Statistics hourly passenger flow 

    group_day=df.groupby(['Date','hour'])['PM'].sum()

    group_day=pd.DataFrame(group_day)

    group_day=group_day.reset_index()

    group_day.hour=group_day.hour.astype('int')

    group_day.sort_values(by=['Date','hour'],inplace=True)

    group_day.reset_index(inplace=True,drop=True)

      

    return group_day



group_day=groupd(df)
#Daily Passenger Flow Drawing

plt.figure(figsize=(12,9),dpi=300)

plt.plot(group_day['PM'])

plt.xlabel(u'day',size=20)

plt.ylabel(u'PM_sum',size=20)

plt.tick_params(labelsize=13)