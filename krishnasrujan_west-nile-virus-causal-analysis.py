import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
data  = pd.read_csv('/kaggle/input/west-nile-mosquitoes/train.csv',parse_dates=['Date'])
data.head(4)
data['date'] = data['Date']
data['year'] = pd.DatetimeIndex(data['Date']).year
data['month'] = [d.strftime('%b') for d in data.Date]
data.set_index('date',inplace=True)
data.shape
data.head(4)
data.groupby('month')['NumMosquitos'].count()
sns.boxplot(x='month', y='NumMosquitos', data=data)
data.groupby('year')['NumMosquitos'].count()
plt.figure(figsize=(20,10))
plt.scatter(y=data[['NumMosquitos']],x=data.index)
plt.show()
data.nunique()
data['Longitude'] = data[['Longitude']].round(2)
data['Latitude'] = data[['Latitude']].round(2)
data.head(4)
data.nunique()
spray  = pd.read_csv('/kaggle/input/west-nile-mosquitoes/spray.csv',parse_dates=['Date'])
spray.head(4)
spray['Longitude'] = spray[['Longitude']].round(2)
spray['Latitude'] = spray[['Latitude']].round(2)
spray.head(4)
spray.nunique()
weather  = pd.read_csv('/kaggle/input/west-nile-mosquitoes/weather.csv',parse_dates=['Date'])
weather.head(4)
weather = weather[['Date','Tavg','Heat','Cool','AvgSpeed']]
weather.ffill()
weather.drop_duplicates(subset='Date',inplace=True)
weather.head(4)
new = pd.merge(data,spray,how='left',on=['Date','Latitude','Longitude'])
new.head(10)
new['spray'] = 1
new.loc[new['Time'].isnull(),'spray'] = 0
new.drop('Time',axis=1,inplace=True)
new.head(4)
print(new.shape)
new = pd.merge(new,weather,how='left',on='Date')
new.shape
new.head(10)
new.set_index('Date',inplace=True)
new.head(4)
plt.figure(figsize=(20,10))
plt.scatter(x=new['Tavg'].sort_values(),y=new['NumMosquitos'])
fig, ax = plt.subplots(figsize=(20,6))
ax.scatter(x=new['AvgSpeed'].sort_values(ignore_index=True),y=new['NumMosquitos'])
labels = ax.get_xticklabels()
plt.setp(labels, rotation=45, horizontalalignment='right')
plt.show()
plt.figure(figsize=(20,10))
plt.scatter(x=new['Heat'].sort_values(),y=new['NumMosquitos'])
plt.figure(figsize=(20,10))
plt.scatter(x=new['Cool'].sort_values(),y=new['NumMosquitos'])
plt.figure(figsize=(20,10))
sns.boxplot(x='month',y='NumMosquitos',data=new,hue='spray')