# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import pandas_profiling 

from scipy.stats import iqr

from sklearn.model_selection import KFold

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV
%matplotlib inline
df = pd.read_csv("../input/uio_clean.csv")
# Data exploratory analysis: Generate profile report from the Quito data set  

pandas_profiling.ProfileReport(df) 
#The variables pickup_datetime and dropoff_datetime need to be set according to Python format

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')

df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'], format='%Y-%m-%d %H:%M:%S')
df.head()

df.shape
#Remove erroneous and missing values

df=df[(df['trip_duration']>df['wait_sec'])]

df.dropna(how='any', axis='rows', inplace=True)
df.shape
np.percentile(df['trip_duration'], np.arange(1,99))
#trip_duration: this value should be 1 minute and 3 hours (many users forget to turn off the taximeter when done with the route)

df=df[df['trip_duration'].between(60, 10800)]

df.shape
#dist_meters: given the size of Quito, 100 km is taken as the upper bound. No negative distancees allowed.

df=df[df['dist_meters'].between(0, 100000)]
df.shape

#wait_sec: must be a positive value and smaller than trip_duration, as mentioned above

df=df[df['wait_sec'].between(0, 10800)]
# To perform a further check for validation of the real duration of the trip, we will check how different are the values trip_duration and (dropoff_datetime' - df'pickup_datetime')

df = df[np.abs(df['trip_duration']- (df['dropoff_datetime'] - df['pickup_datetime']).dt.seconds) <= 2*60]

df.shape
df.head()
#Let´s see the trip duration distribution

plt.hist(df['trip_duration'].values, bins=20)

plt.show()
df.describe()
#Let´s see the distribution of taxi ride providers in Quito

vendor_frequency = df.groupby(['vendor_id'])['id'].count().reset_index()

vendor_frequency
plt.bar(x=vendor_frequency['vendor_id'], height=vendor_frequency['id'])

plt.title("Vendors Distribution ")

plt.xlabel("Vendor Name")

plt.ylabel("No. Trips")

plt.show()
# Let´s transform the variables pickup_datetime and dropoff_datetime, to convert them into numerical variables

df['hour']=df['pickup_datetime'].dt.hour

df['month'] = df['pickup_datetime'].dt.month

df['day_of_week'] = df['pickup_datetime'].dt.dayofweek

df['day_of_year'] = df['pickup_datetime'].dt.dayofyear

df['week_of_year'] = df['pickup_datetime'].dt.weekofyear



#Delete the variables of date time

df=df.drop(['pickup_datetime','dropoff_datetime'], axis=1)
df.head()
#Let's see the distribution of the trips during the months of 2016

x = df.groupby(['month'])['id'].count()

sns.barplot(x.index, x.values)
#Let's see at what times of the day people take more taxis

x = df.groupby(['hour'])['id'].count() 

sns.barplot(x.index, x.values)
#Let's see the distribution of the trips during the days of the week

x = df.groupby(['day_of_week'])['id'].count() 

sns.barplot(x.index, x.values)
#Let's see on what days of the week people spend more time stuck in traffic in Quito

x = df.groupby(['day_of_week'])['wait_sec'].count() 

sns.barplot(x.index, x.values)