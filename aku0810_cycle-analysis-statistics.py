#Importing libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output
%matplotlib inline

import random

import datetime

import pandas as pd

import matplotlib.pyplot as plt

import statistics

import numpy as np

import scipy

from scipy import stats

import seaborn

import warnings

warnings.filterwarnings(action="ignore")
data = pd.read_csv('../input/cycle-share-dataset/trip.csv',error_bad_lines=False)

data.head()
len(data)
data = data.sort_values(by='starttime')

data.reset_index()

print('Date range of dataset: %s - %s' %(data.ix[1, 'starttime'], data.ix[len(data)-1, 'stoptime']))
groupby_user = data.groupby('usertype').size()

groupby_user.plot.bar(title='Distribution of user types ')
groupby_gender = data.groupby('gender').size()

groupby_gender.plot.bar(title='Distribution of Genders ')
#Plotting the distribution of birth years

data = data.sort_values(by='birthyear')

groupby_birthyear = data.groupby('birthyear').size()

groupby_birthyear.plot.bar(title='Distribution of birth years',

figsize = (15,4))
#Plotting the frequency of memeber types for millenials

data_mil = data[(data['birthyear']>=1977) & (data['birthyear']<=1994)]

groupby_mil = data_mil.groupby('usertype').size()

groupby_mil.plot.bar(title='Distribution of user types')
groupby_birthyear_gender = data.groupby(['birthyear','gender'])['birthyear'].count().unstack('gender').fillna(0)

groupby_birthyear_gender[['Male','Female','Other']].plot.bar(title='Distribution of birth years by Gender', stacked=True,

figsize=(15,4))

#Plotting the distribution of birthyears by user types

groupby_birthyear_users = data.groupby(['birthyear','usertype'])['birthyear'].count().unstack('usertype').fillna(0)

groupby_birthyear_users['Member'].plot.bar(title='Distribution of birth years by User Types', stacked=True,

figsize=(15,4))

#Validation if we dont have birth year available for short term pass holders

data[data['usertype'] =='Short-Term Pass Holder']['birthyear'].isnull().values.all()
#Validation If We Don’t Have Gender Available for Short-Term Pass Holders

data[data['usertype']=='Short-Term Pass Holder']['gender'].isnull().values.all()
#Now Time Series Analaysis for the trips 

List_ = list(data['starttime'])

List_ = [datetime.datetime.strptime(x, "%m/%d/%Y %H:%M") for x in List_]

data['starttime_mod'] = pd.Series(List_,index=data.index)

data['starttime_date'] = pd.Series([x.date() for x in List_],index=data.index)

data['starttime_year'] = pd.Series([x.year for x in List_],index=data.index)

data['starttime_month'] = pd.Series([x.month for x in List_],index=data.index)

data['starttime_day'] = pd.Series([x.day for x in List_],index=data.index)

data['starttime_hour'] = pd.Series([x.hour for x in List_],index=data.index)



data.groupby('starttime_date')['tripduration'].mean().plot.bar(title =

'Distribution of Trip duration by date', figsize = (15,4))
trip_duration = list(data['tripduration'])

station_from = list(data['from_station_name'])

print('Mean of trip duration: %f'%statistics.mean(trip_duration))

print('Median of trip duration: %f'%statistics.median(trip_duration))

print('Mode of station originating from: %s'%statistics.mode(station_from))
data['tripduration'].plot.hist(bins=100, title='Frequency distribution of Trip duration')

plt.show()
box = data.boxplot(column=['tripduration'])

plt.show()
q75, q25 = np.percentile(trip_duration, [75,25])

iqr = q75 - q25

print('Proportion of values as outlier: %f percent'%((len(data) - len([x for x in trip_duration if q75 +(1.5*iqr)>=x>=q25-(1.5*iqr)]))*100/float(len(data))))
mean_trip_duration = np.mean([x for x in trip_duration if q75 +(1.5*iqr)>=x>= q25-(1.5*iqr)])

upper_whisker = q75+(1.5*iqr)

print('Mean of trip duration: %f'%mean_trip_duration)



def transform_tripduration(x):

    if x > upper_whisker:

        return mean_trip_duration

    return x

data['tripduration_mean']=data['tripduration'].apply(lambda x: transform_tripduration(x))

data['tripduration_mean'].plot.hist(bins=100, title='Frequency distribution of mean Transformed Trip Duration')

plt.show()

print('Mean of trip duration: %f'%data['tripduration_mean'].mean())

print('Standard deviation of trip duration: %f'%data['tripduration_mean'].std())

print('Median of trip duration: %f'%data['tripduration_mean'].median())

pd.set_option('display.width', 100)

pd.set_option('precision', 3)

data['age'] = data['starttime_year'] - data['birthyear']

correlations = data[['tripduration','age']].corr(method='pearson')

print(correlations)
for cat in ['gender','usertype']:

    print('Category:%s\n'%cat)

    groupby_category = data.groupby(['starttime_date', cat])['starttime_date'].count().unstack(cat)

    groupby_category = groupby_category.dropna()

    category_names = list(groupby_category.columns)



    for comb in [(category_names[i],category_names[j]) for i in range(len(category_names)) for j in range(i+1, len(category_names))]:

        

        



        print('%s %s'%(comb[0], comb[1]))

        t_statistics = stats.ttest_ind(list(groupby_category[comb[0]]),list(groupby_category[comb[1]]))

        print('Statistic: %f, P value: %f'%(t_statistics.statistic,t_statistics.pvalue))

        print('\n')

daily_tickets = list(data.groupby('starttime_date').size())

sample_tickets=[]

checkpoints = [1,10,100,300,500,1000]

plot_count=1

random.shuffle(daily_tickets)

plt.figure(figsize=(15,7))

binrange = np.array(np.linspace(0,700,101))



for i in range(1000):

    if daily_tickets:

        sample_tickets.append(daily_tickets.pop())

    if i+1 in checkpoints or not daily_tickets:

        plt.subplot(2,3,plot_count)

        plt.hist(sample_tickets,binrange)

        plt.title('n=%d' %(i+1),fontsize=15)

        plot_count+=1

    if not daily_tickets:

        break

        



plt.show()

        