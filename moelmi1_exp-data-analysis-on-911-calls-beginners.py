#Start by loading all the libraries we might need .



import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

sns.set_style("whitegrid")

plt.style.use("fivethirtyeight")
df=pd.read_csv('../input/montcoalert/911.csv')
df.head(3)
df.info()

print('--------------------------------------------')

#let's check the top 5 zip codes in the data?

df['zip'].value_counts().head()

#we want top 5 townships for 911 calls

df['twp'].value_counts().head()

#let's make a new column named 'type'extracted from title using lambda func

df['Type']=df['title'].apply(lambda x:x.split(':')[0])
#let's see which type of call is the most.

plt.figure(figsize=(10,6))

sns.countplot(x='Type',data=df, palette='viridis')

plt.title('Type of Calls')


# we will extract new column named (Reasons) to have a clear view of the most reason why people call 911

df['Reason']=df.title.str.split(':',expand=True)[1].str.replace(' -', '')
#let's check our new data after extracting the new columns

df.head(3)

#let's have a look on the Types of the Top 10 Call's Reasons

plt.figure(figsize=(10,6))

sns.countplot(y='Reason',order=df['Reason'].value_counts().index[:10],hue='Type',data=df)

plt.legend(loc=4)

plt.title("Types of the Top 10 Call's Reasons")
#let's have a look an the type of the Calls for Top 10 townships

plt.figure(figsize=(10,8))

sns.countplot(y='twp',order=df['twp'].value_counts().index[:10],hue='Type',data=df)

plt.legend(loc=4)

plt.title("Types of the  Calls in the Top 10 TWP ")
#let's check the top 10 reasons of Emergency Calls in Lower Merion

data1=df[df['twp']=='LOWER MERION']

data2=data1[data1['Type']=='EMS']

plt.figure(figsize=(10,10))

sns.countplot(y='Reason',order=data2['Reason'].value_counts().index[:10],palette='RdYlBu',data=data2)



plt.title('Emergency Calls Reason in LOWER MERION TOWN')
#let's check the top 10 reasons of Emergency Calls in Norristown

data3=df[df['twp']=='NORRISTOWN']

data4=data3[data3['Type']=='EMS']

plt.figure(figsize=(10,10))

sns.countplot(y='Reason',order=data4['Reason'].value_counts().index[:10],palette='RdYlBu',data=data4)



plt.title('Emergency Calls Reason in Norristown')
#we will know go further and try to see which street has most number of calls and the type of the incident using the addr columns 



plt.figure(figsize=(10,12))



sns.countplot(y='addr',data=df,order=df['addr'].value_counts().index[:10],hue='Type',palette='viridis')
#let's have a look what is the main cause of the huge number of EMS Calls from this Street.

plt.figure(figsize=(10,8))

data6=df[df['addr']=='SHANNONDELL DR & SHANNONDELL BLVD']

data7=data6[data6['Type']=='EMS']

sns.countplot(y='Reason',data=data7,order=data7['Reason'].value_counts().index[:10])
df['timeStamp'].dtype
#convert Time stamp column into datetime using pd.to_datetime



df['timeStamp'] = pd.to_datetime(df.timeStamp)
df['timeStamp'].dtype

df['Year'] = df['timeStamp'].dt.year

df['Month'] = df['timeStamp'].dt.month

df['DayOfWeek'] = df['timeStamp'].dt.dayofweek

df['Hour'] = df['timeStamp'].dt.hour
dmap={0:'Mon',1:'Tue',2:'Wed',3:'Thurs',4:'Fri',5:'Sat',6:'Sun'}

df['DayOfWeek'] = df['DayOfWeek'].map(dmap)
df.head()
plt.figure(figsize=(10, 8))

sns.countplot(x=df.DayOfWeek, data=df, hue='Type')

plt.legend(loc=(1,0.1))
#let's  create a countplot of the Year column with the hue based off of the Type column.

plt.figure(figsize=(10, 8))

sns.countplot(x=df.Year, data=df, hue='Type')
#let's have a look on the Top 10 Call's Reasons for last 5 years

plt.figure(figsize=(10,12))

sns.countplot(y='Reason',order=df['Reason'].value_counts().index[:10],hue='Year',data=df)

plt.legend(loc=4)

plt.title("Top 10 Call's Reasons for the last 5 Years")
#plotting hour versus the Types of incident that occurs mostly during the day

plt.figure(figsize=(12,15))

sns.countplot(x='Hour',hue='Type',data=df)
df.head(3)
#lets have a look of how the traffic jams looks during the 12 months in the County

data9=df[df['Type']=='Traffic']

plt.figure(figsize=(8,8))

sns.countplot(x='Month',data=data9)