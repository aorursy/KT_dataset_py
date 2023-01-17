import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

#load data :

file_path = "../input/hotel-booking-demand/hotel_bookings.csv"

data = pd.read_csv(file_path)
data.sample(5)
data.columns
data.shape
data.info()
data.describe()
data.isnull().sum()
null_cols={}

for i in data.columns:

    if data[i].isnull().sum()>0:

        null_cols[i]=data[i].isnull().sum()



        

null_cols
# so as the count of null values in company column is huge so better to drop it

data.drop('company',axis=1,inplace=True)

data.shape
# the next is agent with big number of null values in it.

data['agent'].unique()
data['agent'].mean()
# we will replace the nans with mean.

data['agent'].fillna(value= data['agent'].mean(),inplace=True)

data['agent'].isnull().sum()
data.dropna(how='any',inplace=True)
# Removed or replaces all null values

data.isnull().sum()
# for sake of clearity manually check the data with some damaged data value(e.g. '?',etc)



for i in data.columns:

    print(i+' -> \n',data[i].unique())

    print('\n','-_'*25)
fig,axes = plt.subplots(1,1,figsize=(10,7))

sns.heatmap(data.corr())

plt.show()
num_cols = []

cat_cols = []

for i in data.columns:

    if data[i].dtypes == 'int64' or type(i)=='float64':

        num_cols.append(i)

    else:

        cat_cols.append(i)

print("-"*50)

print("NUMERICAL DATA -> \n",num_cols)

print("-"*50)

print("CATEGORICAL DATA -> \n",cat_cols)
data[num_cols].describe()
fig,axes = plt.subplots(1,1,figsize=(10,7))

sns.heatmap(data[num_cols].corr())

plt.show()
fig,axes = plt.subplots(2,2,figsize=(14,5))

fig.suptitle('Visualization of canceled bookings in different periods', fontsize=16)

sns.barplot(data['arrival_date_year'],data['is_canceled'],ax=axes[0,0])

sns.barplot(data['arrival_date_day_of_month'],data['is_canceled'],ax=axes[0,1])

sns.barplot(data['arrival_date_week_number'],data['is_canceled'],ax=axes[1,0])

sns.barplot(data['lead_time'],data['is_canceled'],ax=axes[1,1])

plt.show()
fig,axes = plt.subplots(3,2,figsize=(18,12))

fig.suptitle('This is visualization of the pattern of "Repeated Guests" ', fontsize=16)

sns.barplot(data['stays_in_week_nights'],data['is_repeated_guest'],ax=axes[0,0])

sns.barplot(data['stays_in_weekend_nights'],data['is_repeated_guest'],ax=axes[0,1])

sns.barplot(data['previous_cancellations'],data['is_repeated_guest'],ax=axes[1,0])

sns.barplot(data['previous_bookings_not_canceled'],data['is_repeated_guest'],ax=axes[1,1])

sns.barplot(data['required_car_parking_spaces'],data['is_repeated_guest'],ax=axes[2,0])

sns.barplot(data['total_of_special_requests'],data['is_repeated_guest'],ax=axes[2,1])

plt.show()