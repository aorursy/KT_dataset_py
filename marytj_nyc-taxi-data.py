# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

#import matplotlib.cm

 

#from mpl_toolkits.basemap import Basemap

import seaborn as sns  # plot styling



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



taxi_train=pd.read_csv('../input/train.csv')

taxi_test=pd.read_csv('../input/test.csv')

#Getting information about the columns

print(taxi_train.columns.values)

print(taxi_test.columns.values)



#Check the data values

#print(taxi_train.head(2))

#Checking datetype of our data columns

print(taxi_train.dtypes)

print(taxi_test.dtypes)



#Checking out missing values in the data

print(taxi_train.isnull().sum())

print(taxi_test.isnull().sum())

#we dont have any missing values in the data 

sns.distplot(taxi_train['trip_duration'])
sns.distplot(np.log(taxi_train['trip_duration']))
#sns.distplot(taxi_train['vendor_id'])

plt.hist(taxi_train['vendor_id'])


taxi_train['pickup_datetime']=pd.to_datetime(taxi_train['pickup_datetime'])

taxi_train['pickup_date'] = taxi_train['pickup_datetime'].dt.date

taxi_train['pickup_time'] = taxi_train['pickup_datetime'].dt.time

#print(taxi_train[['pickup_datetime','pickup_date','pickup_time']])
#plt.plot(taxi_train['pickup_time'],np.log(taxi_train['trip_duration']))

#plt.semilogy(taxi_train['pickup_time'], taxi_train['trip_duration'])

taxi_train['pickup_hour']=taxi_train['pickup_datetime'].dt.hour

print(taxi_train['pickup_hour'].value_counts())

plt.hist(taxi_train['pickup_hour'],bins=24,color='Burlywood',rwidth=0.8)

plt.xlabel('Pick up hour')

#sns.distplot(taxi_train['hour'],bins=24,color='Crimson')
taxi_train['pick_dayofweek']=taxi_train['pickup_datetime'].dt.dayofweek

print(taxi_train['pick_dayofweek'].value_counts())

plt.hist(taxi_train['pick_dayofweek'],bins=24,color='Burlywood',rwidth=0.8)

ax=plt.gca()

b=taxi_train['pick_dayofweek'].unique()

b.sort()

ax.set_xticks(b)

ax.set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])



#speed trip duration according to vendor, also time of the day cross analysis
#speed trip duration according to vendor, also time of the day cross analysis
