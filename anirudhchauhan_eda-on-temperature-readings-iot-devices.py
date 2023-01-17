import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("../input/temperature-readings-iot-devices/IOT-temp.csv")

data.sample(5)
data.head()
data.tail()
print("Shape of our data is : ",data.shape)
print("Unique values in every column \n"+'-'*25)

for i in data.columns:

    print("\t"+i+" = ",len(set(data[i])))
data.info()
data.describe()
df = data.drop(['id','room_id/id'],axis=1)

df.head()
data.isnull().sum()
date=[]

time=[]

for i in df['noted_date']:

    date.append(i.split(' ')[0])

    time.append(i.split(' ')[1])

df['date']=date

df['time']=time
df.drop('noted_date',axis=1,inplace=True)

df.head()
df[['outside','inside']]=pd.get_dummies(df['out/in'])

df.rename(columns = {'out/in':'location'}, inplace = True)
print('Total Inside Observations  :',len([i for i in df['inside'] if  i == 1]))

print('Total Outside Observations :',len([i for i in df['inside'] if  i == 0]))
try:

    df['date'] = pd.to_datetime(df['date'])

    df['year'] = df['date'].dt.year

    df['month'] = df.date.dt.month

    df['day'] = df.date.dt.day

    df.drop('date',axis=1,inplace=True)

except:

    print('Operations already performed')

df.head()
print("Days of observation   : ",sorted(df['day'].unique()))

print("Months of observation : ",sorted(df['month'].unique()))

print("Year of observation   : ",sorted(df['year'].unique()))
print("Temperature -> \n"+"-"*30)

print("\tTotal Count    = ",df['temp'].shape[0])

print("\tMinimum Value  = ",df['temp'].min())

print("\tMaximum Value  = ",df['temp'].max())

print("\tMean Value     = ",df['temp'].mean())

print("\tStd dev Value  = ",df['temp'].std())

print("\tVariance Value = ",df['temp'].var())
df = df[['day','month','year','time','temp','location','outside','inside']]

df.head()
sns.boxplot(df['temp'])

plt.show()
sns.countplot(df['inside'])
sns.barplot(df['location'],df['temp'])

plt.show()
sns.barplot(df['location'],df['temp'])

plt.show()
sns.scatterplot(df['month'],df['temp'],hue=df['inside'])
sns.scatterplot(df['day'],df['temp'],hue=df['inside'])
sns.heatmap(df.corr())
sns.pairplot(df)
arr = df['inside']

x=[]

y=[]

for i in arr:

    if i==1:

        x.append(i)

    else :

        y.append(i)

x=pd.Series(x)

y=pd.Series(y)

type(arr)
fig,axes = plt.subplots(1,3,figsize=(18,5))

sns.violinplot(x,df['temp'],ax=axes[0],color='b').set_title("Inside v/s Temp")

sns.violinplot(y,df['temp'],ax=axes[1],color='r').set_title("Outside v/s Temp")

sns.violinplot(df['location'],df['temp'],ax=axes[2]).set_title("Location v/s Temp")
sns.lineplot(df['day'],df['temp'],hue=df['location'])