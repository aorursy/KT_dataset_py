# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

#test_bikes = pd.read_csv("../input/bike-sharing-system/test_bikes.csv")

df = pd.read_csv("../input/bike-sharing-system/train_bikes.csv")
#quick look at the data

df.tail()
df
df['datetime']= pd.to_datetime(df['datetime']) 
#exploring the data with these visualizing tools

import matplotlib.pyplot as plt

import seaborn as sns
# plotting the counts based on the season

df.plot.scatter(x = 'season', y = 'count')

 # plotting the counts based on the holidays

df.plot.scatter(x = 'holiday', y = 'count')
df.plot.scatter(x = 'workingday', y = 'count')

#relationship between working dat and count
df.plot.scatter(x = 'weather', y = 'count')
df.plot.scatter(x = 'temp', y = 'count')
g = sns.pairplot(df.drop(['datetime'], axis=1),x_vars=["season",

                                                       "holiday",

                                                       "weather",

                                                       "temp",

                                                       "atemp",

                                                       "humidity",

                                                       "windspeed",

                                                       "workingday"],



                 y_vars=["count"],height=10)

#we can generate multiple plots like this using the sns pairplot function 

df.plot.scatter(x = 'windspeed', y = 'count')

#clearly on more windier or stormier days the renting of bikes slows down
df.info()
df.describe() 



# the descriptive statistics that 

# summarize the central tendency,variability 

# and shape of a dataset's distribution
import pandas_profiling

df.profile_report()
#checking of for missing or null values

df.isnull().values.any()
#number of samples, and number of features 

df.shape
df_copy = df.copy()#create a duplicate of the dataset

df_copy = df_copy[ df_copy.datetime.dt.year == 2012 ] #taking just the 2012 info



df_copy.head()
df_copy.loc[:, ('hour')] = df_copy.datetime.dt.hour

df_copy.head()

#now for every entry we have what hour it was taken in and,

#the rest of the information
#using grouby to extract the data

d =df_copy.groupby(['hour', 'season'])['count'].sum().unstack()

d
d.plot(kind='bar', ylim=(0, 80000), figsize=(22,5), width=0.9, title="2012") 
d = df.copy()





d['year'] = d.datetime.dt.year # extratcing the year

d['month'] = d.datetime.dt.month # extratcing the month

#storing year and month info for every sample in a different column 



#then we must group the data based on our chosen structure 

d = d.groupby(['month', 'year'])['count'].sum().unstack()

d.head()

d.plot(kind='bar',  figsize=(15,5), title="2012") 
d = df[df.datetime.dt.year == 2011].copy()

#duplicate our data only using 2011 entries



d['hour'] = d.datetime.dt.hour 

# extratcing the hour info for every entry into a separate column 

d.iloc[779]
#how to generate a list of all the rent counts where hour == 13, ie 1PM

#this will become quite easy as we have 

d[ d.hour == 13 ]['count'].values

hours = {} # this will store a list of values(rent counts) for every hour



for hour in range(24):

        hours[hour] = d[ d.hour == hour ]['count'].values

        #hours[2] will contain all the rent 'count' values for all the entries 

        #that were recorded on 02hrs, 

        

        #this info will be stored in form of a list as oppesed to a single sum value

        #as we saw previously 

hours
plt.figure(figsize=(20,6))

plt.boxplot( [hours[hour] for hour in range(24)] )

axis = plt.gca()

#axis.set_ylim([1, 1100])
d = df[df.season == 1].copy()

#duplicate our data only using season 1 entries 



d['hour'] = d.datetime.dt.hour 

# extratcing the hour info for every entry into a separate column 

d.iloc[2]
hours = {} # this will store a list of values(rent counts) for every hour

for hour in range(24):

        hours[hour] = d[ d.hour == hour ]['count'].values

        #hours[2] will contain all the rent 'count' values for all the entries 

        #that were recorded on 02hrs, 

        

        #this info will be stored in form of a list as oppesed to a single sum value

        #as we saw previously 

hours
plt.figure(figsize=(20,6))

plt.boxplot( [hours[hour] for hour in range(24)] )

axis = plt.gca()

#axis.set_ylim([1, 1100])
d = df.copy()

d['hour'] = d.datetime.dt.hour 

d
d =d.groupby('hour')[['count']].mean()

d.head()
figure,axes = plt.subplots(figsize = (10, 5))

d.plot(kind="line", ax=axes) 
a = df.copy()

a = a.groupby('temp')[['count']].mean()

a.head()
a.plot()
a = df.copy()

a = a.groupby('atemp')[['count']].mean()

a.plot()