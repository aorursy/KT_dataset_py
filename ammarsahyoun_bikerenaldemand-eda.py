import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
df = pd.read_csv('../input/bike-rental-demand/train.csv')
df.head()
df.info()
# To solve datetime of type'object', I will move the datetime to index then to a separate columns. 
df = pd.read_csv('../input/bike-rental-demand/train.csv', parse_dates=['datetime'],index_col=0)# make datetime as an index
df_test = pd.read_csv('../input/bike-rental-demand/test.csv', parse_dates=['datetime'],index_col=0)
df.head()
columns = ['count', 'season', 'holiday', 'workingday', 'weather', 'temp',
       'atemp', 'humidity', 'windspeed', 'year', 'month', 'day', 'dayofweek','hour']
# To deal with datetime column we converted it to index and now will create from index new columns
def add_features(df):
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['dayofweek'] = df.index.dayofweek
    df['hour'] = df.index.hour
add_features(df)
add_features(df_test)
df.head()
plt.title('Rental Gaps') # count distribution 
df['2011-01':'2011-03']['count'].plot()
plt.show()
plt.plot(df['2011-01-01']['count']) # rental distribution hourly
plt.xticks(fontsize=15, rotation=45)
plt.xlabel('Date')
plt.ylabel('Rental count')
plt.title('Hourly rental for Jan 01, 2011')
plt.show()
plt.plot(df['2011-01']['count']) #Seasonal
plt.xticks(fontsize=15, rotation=45)
plt.xlabel('Date')
plt.ylabel('Rental count')
plt.title('Rentals 1 month Jan2011')
plt.show()
group_hour = df.groupby(['hour'])
average_by_hour = group_hour['count'].mean()  #pandas.series
average_by_hour.values
plt.plot(average_by_hour.index, average_by_hour)
plt.xlabel('Hour')
plt.ylabel('Rental count')
plt.xticks(np.arange(24)) # np.array with 24 as our series
plt.grid(True)
plt.title('Average Hourly Rental Count')
plt.plot(df['2011']['count'],label='2011')
plt.plot(df['2012']['count'],label='2012')
plt.xticks(fontsize=14, rotation=45)
plt.xlabel('Date')
plt.ylabel('Rental count')
plt.title('2011 nad 2012 Renatls (year to year)')
plt.legend
plt.show()
group_year_month = df.groupby(['year','month'])
average_year_month = group_year_month['count'].mean()
average_year_month
for year in average_year_month.index.levels[0]:
    plt.plot(average_year_month[year].index, average_year_month[year], label=year)
    
plt.legend()
plt.xlabel('Month')
plt.ylabel('label')
plt.grid(True)
plt.title('Average Monthly Rental Count for 2011, 2012')
plt.show()
group_year_hour = df.groupby(['year','hour'])
average_year_hour = group_year_hour['count'].mean()
for year in average_year_hour.index.levels[0]:
    #print (year)
    #print(average_year_month[year])
    plt.plot(average_year_hour[year].index,average_year_hour[year],label=year)
    
plt.legend()    
plt.xlabel('Hour')
plt.ylabel('Count')
plt.xticks(np.arange(24))
plt.grid(True)
plt.title('Average Hourly Rental Count - 2011, 2012')
group_workingday_hour = df.groupby(['workingday','hour'])
average_workingday_hour = group_workingday_hour['count'].mean()
for workingday in average_workingday_hour.index.levels[0]:
    #print (year)
    #print(average_year_month[year])
    plt.plot(average_workingday_hour[workingday].index,average_workingday_hour[workingday],
             label=workingday)
    
plt.legend()    
plt.xlabel('Hour')
plt.ylabel('Count')
plt.xticks(np.arange(24))
plt.grid(True)
plt.title('Average Hourly Rental Count by Working Day')
plt.show()
df.corr()['count']
# The relation between temperature and rental count?
plt.scatter(x=df.temp,y=df["count"])
plt.grid(True)
plt.xlabel('Temperature')
plt.ylabel('Count')
plt.title('Temperature vs Count')
plt.show()
# The relation between humidity and rental count?
plt.scatter(x=df.humidity,y=df["count"],label='Humidity')
plt.grid(True)
plt.xlabel('Humidity')
plt.ylabel('Count')
plt.title('Humidity vs Count')
plt.show()
np.random.seed(5)
l = list(df.index)
np.random.shuffle(l)
df = df.loc[l]
rows = df.shape[0]
train = int(.7 * rows)
test = rows-train
# check the volume of the test as it was given 3266
rows, train, test
columns
# Write Training Set
df.iloc[:train].to_csv('bike_train.csv'
                          ,index=False,header=False
                          ,columns=columns)
# Write Validation Set
df.iloc[train:].to_csv('bike_validation.csv'
                          ,index=False,header=False
                          ,columns=columns)
# Test Data has only input features
df_test.to_csv('bike_test.csv',index=True,index_label='datetime')
print(','.join(columns))
# Write Column List
with open('bike_train_column_list.txt','w') as f:
    f.write(','.join(columns))