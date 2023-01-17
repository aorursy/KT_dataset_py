#declaring all the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
#reading data and extracting year,weekday,month,hour
df=pd.read_csv('../input/BreadBasket_DMS.csv')
df.dropna(inplace=True)
df['datetime'] = pd.to_datetime(df['Date']+" "+df['Time'])
df['Year'] = df['datetime'].dt.year
df['Month'] = df['datetime'].dt.month
df['Weekday'] = df['datetime'].dt.weekday
df['Hours'] = df['datetime'].dt.hour
df['Day'] = df['datetime'].dt.day


#checking Item data as it is a string
df.groupby(['Item']).groups.keys()
#in above output we found value NONE. So dropping it
df=df[df['Item']!='NONE']
#setting size of plot
plt.figure(figsize=(20,10))
#We can get top 20 sales
df['Item'].value_counts()[:20].plot.bar(title='Top 20 Sales')
#setting size of plot
plt.figure(figsize=(20,10))
#viewing bottom 20 sales
df['Item'].value_counts()[-20:-1].plot.bar(title='Bottom 20 Sales')
#now creating dataframe with integers for checking maximum transaction happened in which hour/month/year/weekday
df1=df[['Transaction', 'Month', 'Year', 'Day','Weekday','Hours']]
df1=df1.drop_duplicates()
plt.figure(figsize=(20,10))
sns.countplot(x='Hours',data=df1)
plt.figure(figsize=(20,10))
sns.countplot(x='Year',data=df1)
plt.figure(figsize=(20,10))
sns.countplot(x='Weekday',data=df1)
plt.figure(figsize=(20,10))
sns.countplot(x='Month',data=df1)
# now moving on how to align items in store. For that we have seen the toppest item soldis coffee.
# So will keep the coffee at a place where people can easily grab it
# Now We have choosen a place for coffee. What about other items.
# Lets say we have a place for coffee. So next item which is ordered maximum with coffee, I will put that item next to it.
# What approach I am following on this goes like this. First find all transactions with minimum 2 items ordered
# So I have two lists one for transaction with more than 1 items and and second with transaction with coffee in it.
#lst1 is prepared by taking intersection of both
df3=df.groupby(['Transaction']).count()
lst1=list(set(df3[df3['Item']>1].index) & set(df[df['Item']=='Coffee']['Transaction']))
#now preparing new dataframe with transaction of coffee and more than 1 item.
df5=pd.DataFrame()
for i in lst1:
    df5=df5.append(df[i==df['Transaction']])
#now plotting graph for what best goes with coffee top 20
plt.figure(figsize=(20,10))
df5[df5['Item']!='Coffee']['Item'].value_counts()[:20].plot.bar(title="What goes with Coffee Best")
#ploting top 20 sales again to compare with above items. The list is different in both
plt.figure(figsize=(20,10))
#We can get top 20 sales
df['Item'].value_counts()[:20].plot.bar(title='Top 20 Sales')
