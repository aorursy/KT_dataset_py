#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#load the train data

train = pd.read_csv('../input/bikesdata/train_bikes.csv', parse_dates=['datetime'])
train.head()
train.plot.scatter(x="season",y= "count");
train.plot.scatter(x = 'holiday', y = 'count');
train.plot.scatter(x = 'workingday', y = 'count');
train.plot.scatter(x = 'temp', y = 'count',color = 'red', alpha = 0.3);
train.plot.scatter(x = 'humidity', y = 'count', color = 'red', alpha = 0.3);# plotting the counts based on humidity
#Using seaborn plot
sns.regplot(x = 'windspeed', y= 'count', data= train);
sns.relplot(x= 'casual' , y = 'count', data= train)
train.info()
#generate descriptive statistics

train.describe()
#Load test data
test = pd.read_csv('../input/bikesdata/test_bikes.csv')
test.head()
#Check the data type and missing values
test.info()
#Generate descriptive statistics
test.describe()
df = train.copy()
df["hour"] = df.datetime.dt.hour
bb_hour = df.groupby(['hour', 'workingday'])['count'].sum().unstack()
bb_hour.plot(kind='bar', figsize=(15,5), width=0.9, title="Year").legend()
#function to plot no. of bikes rented
def rent_bike_plot(variable):
    #extract times
    #df["year"] = df.datetime.dt.year
    #df["month"] = df.datetime.dt.month
    
    bb_plot = df.groupby([variable, "year"])['count'].sum().unstack()
    return bb_plot.plot(kind= "bar", figsize = (15,5), width=0.9, title = "Bikes rented per {} in 2011 and 2012".format(variable))
    
rent_bike_plot("hour");
hours = {}
for hour in range(24):
    hours[hour] = df[ df.hour == hour ]['count'].values
fig, ax = plt.subplots(figsize = (10,5))                       
avg_count = df.groupby(["hour"])["count"].agg("mean")

avg_count.plot(kind="line", color="green", ax=ax)
plt.title("Avg bike rented vs hour")
plt.xlabel("Time in hours")
plt.ylabel("Average bikes rented")
plt.show()
# plotting temp vs count

temp = df.groupby('temp')['count'].mean()
temp[:5]
temp.plot(figsize = (15,5), color="green")
plt.title("Temperature vs bikes rented")
plt.ylabel('# of bikes rented')
plt.show()