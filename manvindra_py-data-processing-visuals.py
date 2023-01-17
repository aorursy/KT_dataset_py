# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import datetime

import calendar

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

pd.set_option('display.float_format', lambda x: '%.2f' % x)



# Any results you write to the current directory are saved as output.
#Load all Files

data_load = {

    'air_reserve': pd.read_csv('../input/air_reserve.csv',parse_dates=['visit_datetime','reserve_datetime']), 

    'hpg_reserve': pd.read_csv('../input/hpg_reserve.csv',parse_dates=['visit_datetime','reserve_datetime']), 

    'air_store': pd.read_csv('../input/air_store_info.csv'),

    'hpg_store': pd.read_csv('../input/hpg_store_info.csv'),

    'air_visit': pd.read_csv('../input/air_visit_data.csv',parse_dates=['visit_date']),

    'store_id': pd.read_csv('../input/store_id_relation.csv'),

    'sample_sub': pd.read_csv('../input/sample_submission.csv'),

    'holiday_dates': pd.read_csv('../input/date_info.csv',parse_dates=['calendar_date']).rename(columns={'calendar_date':'visit_date'})

    }
# Air Reserve: reservations made in the air system

data_load['air_reserve'].head()

# std	4.92 # min	1.00 # 25%	2.00 # 50%	3.00 # 75%	5.00 # max	100.00
#hpg_reserve: reservations made in the hpg system

data_load['hpg_reserve'].head()

# mean	5.07 # std	5.42 # min	1.00 # 25%	2.00 # 50%	3.00 # 75%	6.00 # max	100.00
#air_store: information about select air restaurants

data_load['air_store'].describe(include = ['O'])
#hpg_store: information about select hpg restaurants

data_load['hpg_store'].describe(include = ['O'])
#air_visit : historical visit data for the air restaurants.

data_load['air_visit'].head()
#holiday_dates: basic information about the calendar dates in the dataset.

data_load['holiday_dates'].head(5)
# Submission only contains Air id

data_load['sample_sub'].head(1)
# Removing concat from air id and date

data_load['sample_sub']['air_store_id'] = data_load['sample_sub']['id'].apply(lambda x: '_'.join(x.split('_')[:2]))

data_load['sample_sub']['visit_date'] = data_load['sample_sub']['id'].apply(lambda x: x.split('_')[-1])
data_load['sample_sub'].head(1)
#Visualization libs

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import gridspec

%matplotlib inline

from datetime import datetime
#Visitor each day

plt1 = data_load['air_visit'].groupby(['visit_date'], as_index=False).agg({'visitors': np.sum})

plt1=plt1.set_index('visit_date')

plt1.plot(figsize=(15, 6))

plt.ylabel("Sum of Visitors")

plt.title("Visitor each day")
# Pax Frequency: Count of visit with 'x' visitor

plt2=data_load['air_visit']['visitors'].value_counts().reset_index().sort_index()

fig,ax = plt.subplots()

ax.bar(plt2['index'] ,plt2['visitors'])

fig.set_size_inches(15,4, forward=True)

ax.set_title("PAX Frequency")

ax.set_ylabel('Counts')

ax.set_xlabel('Number of People in a visit')
#Median number of visitor in day of a week

data_load['air_visit']['dow']=data_load['air_visit']['visit_date'].apply(lambda x: calendar.day_name[x.weekday()])

plt3 = data_load['air_visit'].groupby(['dow'], as_index=False).agg({'visitors': np.median})

days = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']

mapping = {day: i for i, day in enumerate(days)}

key = plt3['dow'].map(mapping)

plt3 = plt3.iloc[key.argsort()].set_index('dow').reset_index()



#Median number of visitor in Month of a Year

data_load['air_visit']['Month']=data_load['air_visit']['visit_date'].apply(lambda x: calendar.month_name[x.month])

plt4 = data_load['air_visit'].groupby(['Month'], as_index=False).agg({'visitors': np.median})

Months = ['January','February','March','April','May','June','July','August','September','October','November','December']

mapping = {Month: i for i, Month in enumerate(Months)}

key = plt4['Month'].map(mapping)

plt4 = plt4.iloc[key.argsort()].set_index('Month').reset_index()

#plot

fig, ax =plt.subplots(1,2)

fig.set_size_inches(15,4, forward=True)



sns.barplot(x="dow",y="visitors",data=plt3,ax=ax[0])

sns.barplot(x="Month",y="visitors",data=plt4,ax=ax[1])

ax[0].set_xlabel('Day of week')

ax[0].set_ylabel('Median Visitors')

ax[1].set_ylabel('Median Visitors')

for ax in ax:

    for label in ax.get_xticklabels():

        label.set_rotation(45) 
# Obesrvation pending
data_load['air_reserve'].head()
# Compare Reservation data to Visitor data

#Visitor each day

data_load['air_reserve']['visit_date']=data_load['air_reserve']['visit_datetime'].apply(lambda x: x.date())

data_load['air_reserve']['reserve_date']=data_load['air_reserve']['reserve_datetime'].apply(lambda x: x.date())

airR1 = data_load['air_reserve'].groupby(['visit_date'], as_index=False).agg({'reserve_visitors': np.sum})

airR1=airR1.set_index('visit_date')

airR1.plot(figsize=(15, 6))

plt.ylabel("Sum of Visitors")

plt.title("Visitor each day")
data_load['air_reserve']['visit_hr']=data_load['air_reserve']['visit_datetime'].apply(lambda x: x.time().hour)

data_load['air_reserve']['reserve_hr']=data_load['air_reserve']['reserve_datetime'].apply(lambda x: x.time().hour)

data_load['air_reserve']['diff_hr']=(data_load['air_reserve']['visit_datetime']-data_load['air_reserve']['reserve_datetime']).apply(lambda x : x.total_seconds()/3600)

airR2 = data_load['air_reserve'].groupby(['visit_hr'], as_index=False).agg({'reserve_visitors': np.sum})

airR3 = data_load['air_reserve'].groupby(['diff_hr'], as_index=False).agg({'reserve_visitors': np.sum})
#plot

fig = plt.figure(figsize=(15, 6)) 

gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 2.5]) 

ax0 = plt.subplot(gs[0])

ax1 = plt.subplot(gs[1])

sns.barplot(x="visit_hr",y="reserve_visitors",data=airR2,ax=ax0)

sns.barplot(x="diff_hr",y="reserve_visitors",data=airR3[(airR3['diff_hr'] <= 50)],ax=ax1)

ax0.set_xlabel('Visitor')

ax0.set_ylabel('Sum of Reserve Visitor')

ax1.set_ylabel('Sum of Reserve Visitor')

for ax in [ax0,ax1]:

    for label in ax.get_xticklabels():

        label.set_rotation(90) 
# Compare Reservation data to Visitor data

#Visitor each day

data_load['hpg_reserve']['visit_date']=data_load['hpg_reserve']['visit_datetime'].apply(lambda x: x.date())

data_load['hpg_reserve']['reserve_date']=data_load['hpg_reserve']['reserve_datetime'].apply(lambda x: x.date())

hpgR1 = data_load['hpg_reserve'].groupby(['visit_date'], as_index=False).agg({'reserve_visitors': np.sum})

hpgR1=hpgR1.set_index('visit_date')

hpgR1.plot(figsize=(15, 6))
data_load['hpg_reserve']['visit_hr']=data_load['hpg_reserve']['visit_datetime'].apply(lambda x: x.time().hour)

data_load['hpg_reserve']['reserve_hr']=data_load['hpg_reserve']['reserve_datetime'].apply(lambda x: x.time().hour)

data_load['hpg_reserve']['diff_hr']=(data_load['hpg_reserve']['visit_datetime']-data_load['hpg_reserve']['reserve_datetime']).apply(lambda x : x.total_seconds()/3600)

hpgR2 = data_load['hpg_reserve'].groupby(['visit_hr'], as_index=False).agg({'reserve_visitors': np.sum})

hpgR3 = data_load['hpg_reserve'].groupby(['diff_hr'], as_index=False).agg({'reserve_visitors': np.sum})

#plot

fig = plt.figure(figsize=(15, 6)) 

gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 2.5]) 

ax0 = plt.subplot(gs[0])

ax1 = plt.subplot(gs[1])

sns.barplot(x="visit_hr",y="reserve_visitors",data=hpgR2,ax=ax0)

sns.barplot(x="diff_hr",y="reserve_visitors",data=hpgR3[(hpgR3['diff_hr'] <= 50)],ax=ax1)

ax0.set_xlabel('Visitor')

ax0.set_ylabel('Sum of Reserve Visitor')

ax1.set_ylabel('Sum of Reserve Visitor')

for ax in [ax0,ax1]:

    for label in ax.get_xticklabels():

        label.set_rotation(90) 
# Number of restaurent in area: Air Data

airS1=data_load['air_store']['air_area_name'].value_counts().reset_index().sort_index()

airS2=data_load['air_store']['air_genre_name'].value_counts().reset_index().sort_index()

fig,ax = plt.subplots(1,2)

sns.barplot(y='index' ,x='air_area_name',data=airS1.iloc[:15],ax=ax[0])

sns.barplot(y='index' ,x='air_genre_name',data=airS2.iloc[:15],ax=ax[1])

fig.set_size_inches(15,10, forward=True)

ax[0].set_ylabel('Number of Restaurent')

ax[1].set_ylabel('Number of Restaurent')
# Number of restaurent in area: HPG store

hpgS1=data_load['hpg_store']['hpg_area_name'].value_counts().reset_index().sort_index()

hpgS2=data_load['hpg_store']['hpg_genre_name'].value_counts().reset_index().sort_index()

fig,ax = plt.subplots(1,2)

sns.barplot(y='index' ,x='hpg_area_name',data=hpgS1.iloc[:15],ax=ax[0])

sns.barplot(y='index' ,x='hpg_genre_name',data=hpgS2.iloc[:15],ax=ax[1])

fig.set_size_inches(15,10, forward=True)

ax[0].set_ylabel('Number of Restaurent')

ax[1].set_ylabel('Number of Restaurent')