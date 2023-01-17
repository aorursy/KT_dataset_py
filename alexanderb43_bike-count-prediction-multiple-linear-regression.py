import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm

%matplotlib inline
df = pd.read_csv('../input/train.csv')
df.rename(columns = {'count':'Total'}, inplace = True)
df.head(3)
df['datetime'] = pd.to_datetime(df.datetime)
df['Year'] = df.datetime.dt.year

df['Month'] = df.datetime.dt.month

df['Day'] = df.datetime.dt.day

df['time'] = df.datetime.dt.time

df['hour']=[t.hour for t in pd.DatetimeIndex(df.datetime)]
df['Mon_Year'] = df.Month.map(str) + '-' + df.Year.map(str)

df['Day_Mon_Year'] = df.Day.map(str) + '-' + df.Month.map(str) + '-' + df.Year.map(str)
df.groupby(['time']).Total.mean().plot(color = 'red')

plt.locator_params(nbins=7)

plt.grid()

plt.title('Average Number of Bikes Rode by Time')

plt.xlabel('Time of Day')

plt.ylabel('Average Number of Bikes')

plt.style.use('default')
df.groupby(['Month']).casual.sum().plot(color = 'red')

df.groupby(['Month']).registered.sum().plot(color = 'purple')

plt.yticks([50000,100000,150000,200000],['50,000','100,000','150,000','200,000'])

plt.title('Number of Bikes Rode by Month From 2011-2012')

plt.ylabel('Number of Bikes')

plt.grid()

plt.legend()
df.groupby(['datetime']).Total.sum().sort_values(ascending = False).head().plot.bar(color = 'blue')

plt.ylim(930,980)

plt.title('Highest Number of Bikes Rode')

plt.xlabel('Time')

plt.ylabel('Number of Bikes')
df.groupby(['Year']).registered.sum().plot.bar(color = 'grey')

df.groupby(['Year']).casual.sum().plot.bar(color = 'red')

plt.title('Yearly Comparsion of Registered Vs Casual Users')

plt.xlabel('Year')

plt.ylabel('Number of Bikes')

plt.yticks([200000,400000,600000,800000,1000000],['200,000','400,000','600,000','800,000','1,000,000'])

plt.legend()
df.groupby(['weather']).Total.sum().plot.bar()

plt.title('Total Bikes Rode during a specific weather')

plt.xlabel('Weather')

plt.ylabel('Number of Bikes Rode')

plt.style.use('default')

plt.xticks([0,1,2,3],['Clear','Cloudy','Downfall','Heavy Downfall'])

plt.yticks([0,200000,400000,600000,800000,1000000,1200000,1400000],['0','200k','400k','600k','800k','1 Mil','1.2 Mil','1.4 Mil'])
df.groupby(df['workingday']).Total.sum().plot.pie(autopct = '%.1f%%',labels = ['Weekend','Weekday'],colors = ['lightblue','lightgreen'],shadow=True)

plt.title('Percentage of Bikes Rode during the Weekday vs Weekend')

plt.ylabel('')
df.groupby(['season']).Total.sum().sort_values(ascending = False).head().plot.pie(autopct='%.1f%%',colors=['lightblue','darkred','lightgreen','gray'],

                                                                                 labels = ['Fall','Summer','Winter','Spring'],shadow=True)

plt.ylabel('')

plt.title('Percentage Breakdown of Bikes rode during a season')
df.groupby(['Month']).Total.sum().sort_values(ascending = False).head().plot.pie(autopct='%.1f%%',colors=['lightblue','darkred','lightgreen','gray','purple'],

                                                                                 labels =['Jun','Jul','Aug','Sep','Oct'],shadow=True)

plt.ylabel('')

plt.title('Percentage Breakdown of Top Five Months')
plt.scatter(x=df['temp'],y=df['Total'])

plt.title('Temperature (C) vs Total Bikes Rode')

plt.xlabel('Temperature (C)')

plt.ylabel('Number of Bikes Rode')

plt.style.use('default')
plt.scatter(x=df['humidity'],y=df['Total'])

plt.title('Humidity vs Total Bikes Rode')

plt.xlabel('Humidty')

plt.ylabel('Number of Bikes Rode')

plt.xticks()

plt.yticks()

plt.scatter(x=df['windspeed'],y=df['Total'])

plt.title('Windspeed vs Total Bikes Rode')

plt.xlabel('Windspeed')

plt.ylabel('Number of Bikes Rode')

df.corr()
X = df.ix[:,[1,2,3,4,5,7,8,9,16]]

Y = df.Total

Xc= sm.add_constant(X)
Regress = sm.OLS(Y,Xc)

FitMod = Regress.fit()

FitMod.summary()
df['Predicted'] = FitMod.predict(Xc)

df['Predicted'] = abs(df.Predicted)

Res = df.Total - df.Predicted

df['Residuals'] = Res
df.groupby(['Month']).Total.sum().plot(color = 'blue')

df.groupby(['Month']).Predicted.sum().plot(color = 'yellow')

plt.title('Actual Vs Predicted Bike Totals by Month')

plt.ylabel('Number of Bikes')

plt.yticks([80000,100000,120000,140000,160000,180000,200000,220000],['80K','100K','120K','140K','160K','180K',

                                                                     '200K','220K'])

plt.grid()

plt.legend()
df.groupby(['hour']).Total.mean().plot(color = 'red')

df.groupby(['hour']).Predicted.mean().plot(color = 'purple')

plt.title('Average Bikes Rode by Hour')

plt.ylabel('Average number of Bikes')

plt.locator_params(nbins=3)

plt.grid()

plt.legend(labels = ['Actual','Predicted'])