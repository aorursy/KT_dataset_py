import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#Gets rid of the seaborn 'FutureWarning'
steps = pd.read_csv('../input/com.samsung.shealth.step_daily_trend.201812071026.csv', header=1 ) 
steps.head()
steps['day_time'] = pd.to_datetime(steps['day_time'], unit='ms')
steps.head()
steps = steps[steps['source_type']==0].groupby('day_time').sum()
steps.drop(['source_type','speed', 'calorie'], axis=1, inplace=True)
steps.reset_index(inplace=True)
steps.head()
steps['Day'] = steps['day_time'].apply(lambda datestamp: datestamp.day)
steps['Year'] = steps['day_time'].apply(lambda datestamp: datestamp.year)


#Functions to get days of the week and months as strings instead of indexes 
#(0-6 and 0-12 respectively)

def dayofweek(datestamp):
    return ['Mon', 'Tue', 'Wed','Thur','Fri','Sat','Sun'][datestamp.weekday()]
def monthname(datestamp):
    return ['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][datestamp.month-1]

steps['Weekday'] = steps['day_time'].apply(lambda datestamp: dayofweek(datestamp)) 
steps['MonthName'] = steps['day_time'].apply(lambda datestamp: monthname(datestamp))

#Keeping the month as an index to construct the yearmonth column below:

steps['Month'] = steps['day_time'].apply(lambda datestamp: datestamp.month)

#Function to get a combined 'YearMonth' column

def yearmonth(cols):
    month=cols[0]
    year=cols[1]
    return '{}-{}'.format(month, year)

steps['YearMonth'] = steps[['Month', 'Year']].apply(lambda cols: yearmonth(cols), axis=1)

steps.head()
steps.drop('day_time', inplace=True, axis=1)

steps.head()
print ('Number of years = {}'.format(steps['count'].count()/365))
steps['count'].describe()
steps[steps['count']==steps['count'].max()][['Day','MonthName', 'Year']]
sns.distplot(steps['count'], bins=25).set(xlim=(0,steps['count'].max()))  
plt.figure(figsize=(20,5))
plt.tight_layout()
plt.title('Average steps per day in 2017 and 2018')
sns.barplot(x='Year', y=steps['count'], data=steps[steps['Year']>2016])
plt.figure(figsize=(20,5))
plt.tight_layout()
plt.title('Average steps per day for each month, 2017 vs 2018')
sns.barplot(x='MonthName', y='count', data=steps[steps['Year']>2016], hue='Year', order = ['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] )
plt.figure(figsize=(20,5))
plt.tight_layout()
plt.title('Average steps per day over a typical month')
sns.barplot(x='Day', y='count', data=steps)
plt.figure(figsize=(20,5))
plt.tight_layout()
plt.title('Average steps per day over a typical week')
sns.barplot(x='Weekday', y='count', data=steps, order = ['Mon', 'Tue', 'Wed','Thur','Fri','Sat','Sun'], palette = 'deep')
piv = steps[steps['Year']==2017].pivot_table(index='Month',columns='Day', values='count').fillna(0)
plt.figure(figsize=(20,5))
plt.title('Steps in 2017')
sns.heatmap(piv, cmap='viridis')
plt.figure(figsize=(10,5))
plt.title('Total Distance (km)')
sns.barplot(x='Year', y=steps['distance']/1000, data=steps[steps['Year']>2016], estimator= sum)
plt.figure(figsize=(10,5))
plt.title('Total Steps (in millions)')
sns.barplot(x='Year', y=steps['count']/1000000, data=steps[steps['Year']>2016], estimator= sum)
sns.scatterplot(x='count', y='distance', data=steps)
from sklearn.linear_model import LinearRegression  #For step size regression fit

lm= LinearRegression()
x=steps['count'].values.reshape(-1, 1)
y=steps['distance'].values.reshape(-1, 1)
lm.fit(x,y)
print('Average step size = {:.2f} cm'.format(lm.coef_[0][0]*100))
total_distance = steps['distance'].sum()
total_steps =  steps['count'].sum()
average_step = total_distance/total_steps
print('Average step size = {:.2f} cm'.format(average_step))
thresh=1000
lazydays= steps[steps['count']<thresh]['count'].count()
print('{} days where steps < {}'.format(lazydays,thresh))
def at_least(n):
    return steps[steps['count']>n]['count'].count()

print (at_least(10000))
print('Laziest day = {}'.format(steps['count'].min()))
print('Busiest day = {}'.format(steps['count'].max()))
sns.lineplot(x= steps['count'], y= steps['count'].apply(lambda values : at_least(values)))
