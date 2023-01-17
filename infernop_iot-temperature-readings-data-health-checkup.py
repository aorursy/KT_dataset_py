# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt

import matplotlib.pyplot as plt 

%matplotlib inline 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("/kaggle/input/temperature-readings-iot-devices/IOT-temp.csv")

display(dataset.head())

print("Shape of Dataset : ", dataset.shape)
dataset.dtypes
dataset["room_id/id"].nunique()

dataset.drop(columns=['room_id/id','id'], inplace=True)
dataset.rename(columns={'out/in':"Out_In"}, inplace=True)
print("Before duplicate treatment : ", dataset.shape)

dataset.drop_duplicates(subset=['noted_date','Out_In','temp'], keep='first',inplace=True)

print("After duplicate treatment : ", dataset.shape)
print("Data recorded from {} to {}".format(min(dataset.noted_date), max(dataset.noted_date)))

print("Minimum Temperature : {} degrees \nMaximum Temperature : {} degrees".format(min(dataset.temp), max(dataset.temp)))
dataset['date'] = dataset['noted_date'].apply(lambda x: dt.datetime.strptime(x, '%d-%m-%Y %H:%M').date())





dataset['day'] = dataset['noted_date'].apply(lambda x: dt.datetime.strptime(x, '%d-%m-%Y %H:%M').day)

dataset['month'] = dataset['noted_date'].apply(lambda x: dt.datetime.strptime(x, '%d-%m-%Y %H:%M').month)

dataset['year'] = dataset['noted_date'].apply(lambda x: dt.datetime.strptime(x, '%d-%m-%Y %H:%M').year)



dataset['hour'] = dataset['noted_date'].apply(lambda x: dt.datetime.strptime(x, '%d-%m-%Y %H:%M').hour)

dataset['minute'] = dataset['noted_date'].apply(lambda x: dt.datetime.strptime(x, '%d-%m-%Y %H:%M').minute)

dataset.drop(columns = ['noted_date'], inplace = True)

dataset.head()
grouped_by_date = pd.DataFrame(dataset.groupby('date')['minute'].count())

min_freq = grouped_by_date.min()

max_freq = grouped_by_date.max()

grouped_by_date.reset_index(inplace=True)

print(type(grouped_by_date))

display(grouped_by_date.head())
grouped_by_date.rename(columns = {"date": "date", 

                                  "minute":"frequency"}, inplace = True)

grouped_by_date.head()

plt.figure(figsize=(5,9))

_ = plt.boxplot(grouped_by_date.frequency)
def ioflag(x):

    if x=='In':

        return 1

    else:

        return 0

dataset['IO_Flag'] = dataset['Out_In'].apply(lambda x: ioflag(x))

dataset.head()
print("Number of records from dataset for Out : ", dataset[dataset['Out_In'] == 'Out'].shape[0]

,"\nNumber of records from dataset for In :  ",dataset[dataset['Out_In'] == 'In'].shape[0])

print("% Distribution for Outside/Inside is 73:28 ")#, 100*(dataset[dataset['Out_In'] == 'Out'].shape[0]/dataset.shape[0]))
daily_measurement_frequency = pd.DataFrame(dataset.groupby(['date','Out_In'])['temp'].mean())

daily_measurement_frequency.reset_index(inplace=True)

daily_measurement_frequency.rename(columns={'temp':'mean_temperature'}, inplace = True)

daily_measurement_frequency.head()
print(type(daily_measurement_frequency))
daily_measurement_frequency[daily_measurement_frequency['mean_temperature'] <= 1000 ].head(20)
colors = {'In':'red', 'Out':'blue'}

plt.figure(figsize=(15,6))

plt.xticks(rotation=70)

plt.scatter(daily_measurement_frequency['date'],

        daily_measurement_frequency['mean_temperature'],

        c=daily_measurement_frequency['Out_In'].apply(lambda x : colors[x]))

plt.show()
daily_measurement_variety = pd.DataFrame(dataset[['date','Out_In']].groupby(['date'])['Out_In'].nunique())

daily_measurement_variety.reset_index(inplace=True)

daily_measurement_variety.rename(columns = {'Out_In':'Measurements'}, inplace=True)

daily_measurement_variety.head()
plt.figure(figsize=(15,6))

plt.xticks( rotation=70)

plt.yticks(np.arange(0,5,1.0))

plt.scatter(daily_measurement_variety['date'],daily_measurement_variety['Measurements'] )
print(daily_measurement_variety['Measurements'].value_counts())
dataset.date.nunique()
dataset = dataset.merge(daily_measurement_variety, left_on='date', right_on='date')
dataset.head()
dataset.groupby(['Measurements','Out_In']).count()
dataset.groupby(['Measurements','Out_In'])['date'].nunique()
dataset.shape
monthly_split = pd.DataFrame(dataset.groupby(['month'])['temp'].mean())

monthly_split.reset_index(inplace=True)

monthly_split.rename(columns={'temp':'mean_temp'})

monthly_split.head()
display(monthly_split.sort_values(by=['temp'], ascending=False))

display(monthly_split.sort_values(by=['temp'], ascending=True))

monthly_type_split = pd.DataFrame(dataset.groupby(['month','Out_In',])['temp'].mean())

monthly_type_split.reset_index(inplace=True)

monthly_type_split.rename(columns={'temp':'mean_temp'})

monthly_type_split.head()
display(monthly_type_split[monthly_type_split['Out_In']=='In'].sort_values(by=['temp'], ascending=True))

display(monthly_type_split[monthly_type_split['Out_In']=='Out'].sort_values(by=['temp'], ascending=True))



display(monthly_type_split[monthly_type_split['Out_In']=='In'].sort_values(by=['temp'], ascending=False))

display(monthly_type_split[monthly_type_split['Out_In']=='Out'].sort_values(by=['temp'], ascending=False))