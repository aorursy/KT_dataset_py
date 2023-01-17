#importing necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.patches import Circle



#setting plot style to seaborn

plt.style.use('seaborn')
#reading data

df = pd.read_csv('../input/citibike-system-data/201306-citibike-tripdata.csv')

df.head()
#we have 5,77,703 rows and 15 columns, seems to be quite a bit of missing values

df.info()
#sum of missing values in each column

df.isna().sum()
#calculating the percentage of missing values

#sum of missing value is the column divided by total number of rows in the dataset multiplied by 100



data_loss1 = round((df['end station id'].isna().sum()/df.shape[0])*100)

data_loss2 = round((df['birth year'].isna().sum()/df.shape[0])*100)



print(data_loss1, '% of data loss if NaN rows of end station id, \nend station name, end station latitude and end station longitude dropped.\n')

print(data_loss2, '% of data loss if NaN rows of birth year dropped.')
#dropping NaN values

rows_before_dropping = df.shape[0]



#droppping missing valued rows from birth year will a loss of 42% of data,

#so decided to drop entire birth year column.

df.drop('birth year',axis=1, inplace=True)



#Now left with end station id, end station name, end station latitude and end station longitude

#these four columns have missing values in exact same row,

#so dropping NaN from all four columns will still result in 3% data loss

df.dropna(axis=0, inplace=True)

rows_after_dropping = df.shape[0]



#total data loss

print('% of data lost: ',((rows_before_dropping-rows_after_dropping)/rows_before_dropping)*100)



#checking for NaN

df.isna().sum()
#plotting total no.of males and females

splot = sns.countplot('gender', data=df)



#adding value above each bar:Annotation

for p in splot.patches:

    an = splot.annotate(format(p.get_height(), '.2f'),

                        #bar value is nothing but height of the bar

                       (p.get_x() + p.get_width() / 2., p.get_height()),

                       ha = 'center',

                       va = 'center', 

                       xytext = (0, 10), 

                       textcoords = 'offset points')

    an.set_size(20)#test size

splot.axes.set_title("Gender distribution",fontsize=30)

splot.axes.set_xlabel("Gender",fontsize=20)

splot.axes.set_ylabel("Count",fontsize=20)



#adding x tick values

splot.axes.set_xticklabels(['Unknown', 'Male', 'Female'])

plt.show()
#number of subscribers(annual pass) vs customers(24 hours/3day pass)

user_type_count = df['usertype'].value_counts()

plt.pie(user_type_count.values, labels=user_type_count.index ,autopct='%1.2f%%', textprops={'fontsize': 15} )

plt.title('Subscribers vs Customers', fontsize=20)

plt.show()
#converting trip duration from seconds to minuits

df['tripduration'] = df['tripduration']/60



#creating bins (0-30min, 30-60min, 60-120min, 120 and above)

max_limit = df['tripduration'].max()

df['tripduration_bins'] = pd.cut(df['tripduration'], [0, 30, 60, 120, max_limit])



sns.barplot(x='tripduration_bins', y='tripduration', data=df, estimator=np.size)

plt.title('Usual trip duration', fontsize=30)

plt.xlabel('Trip duration group', fontsize=20)

plt.ylabel('Trip Duration', fontsize=20)

plt.show()
#number of trips that started and ended at same station

start_end_same = df[df['start station name'] == df['end station name']].shape[0]



#number of trips that started and ended at different station

start_end_diff = df.shape[0]-start_end_same



fig,ax=plt.subplots()

ax.pie([start_end_same,start_end_diff], labels=['Same', 'Different'], autopct='%1.2f%%', textprops={'fontsize': 20})

ax.set_title('Same start and end location vs Different start and end location', fontsize=20)





circle = Circle((0,0), 0.6, facecolor='white')

ax.add_artist(circle)



plt.show()
#converting string to datetime object

df['starttime']= pd.to_datetime(df['starttime'])



#since we are dealing with single month, we grouping by days

#using count aggregation to get number of occurances i.e, total trips per day

start_time_count = df.set_index('starttime').groupby(pd.Grouper(freq='D')).count()



#we have data from July month for only one day which is at last row, lets drop it

start_time_count.drop(start_time_count.tail(1).index, axis=0, inplace=True)



#again grouping by day and aggregating with sum to get total trip duration per day

#which will used while plotting

trip_duration_count = df.set_index('starttime').groupby(pd.Grouper(freq='D')).sum()



#again dropping the last row for same reason

trip_duration_count.drop(trip_duration_count.tail(1).index, axis=0, inplace=True)



#plotting total rides per day

#using start station id to get the count

fig,ax=plt.subplots(figsize=(25,10))

ax.bar(start_time_count.index, 'start station id', data=start_time_count, label='Total riders')

#bbox_to_anchor is to position the legend box

ax.legend(loc ="lower left", bbox_to_anchor=(0.01, 0.89), fontsize='20')

ax.set_xlabel('Days of the month June 2013', fontsize=30)

ax.set_ylabel('Riders',  fontsize=40)

ax.set_title('Bikers trend for the month June', fontsize=50)



#creating twin x axis to plot line chart is same figure

ax2=ax.twinx()

#plotting total trip duration of all user per day

ax2.plot('tripduration', data=trip_duration_count, color='y', label='Total trip duration', marker='o', linewidth=5, markersize=12)

ax2.set_ylabel('Time duration',  fontsize=40)

ax2.legend(loc ="upper left", bbox_to_anchor=(0.01, 0.9), fontsize='20')



ax.set_xticks(trip_duration_count.index)

ax.set_xticklabels([i for i in range(1,31)])



#tweeking x and y ticks labels of axes1

ax.tick_params(labelsize=30, labelcolor='#eb4034')

#tweeking x and y ticks labels of axes2

ax2.tick_params(labelsize=30, labelcolor='#eb4034')



plt.show()
#top 10 start station

top_start_station = df['start station name'].value_counts()[:10]



fig,ax=plt.subplots(figsize=(20,8))

ax.bar(x=top_start_station.index, height=top_start_station.values, color='#70eb67', width=0.5)



#adding value above each bar:Annotation

for p in ax.patches:

    an = ax.annotate(format(p.get_height(), '.2f'), 

                   (p.get_x() + p.get_width() / 2., p.get_height()), 

                   ha = 'center',

                   va = 'center', 

                   xytext = (0, 10), 

                   textcoords = 'offset points')

    an.set_size(20)

ax.set_title("Top 10 start locations in NY",fontsize=30)

ax.set_xlabel("Station name",fontsize=20)



#rotating the x tick labels to 45 degrees

ax.set_xticklabels(top_start_station.index, rotation = 45, ha="right")

ax.set_ylabel("Count",fontsize=20)

#tweeking x and y tick labels 

ax.tick_params(labelsize=15)

plt.show()
#top 10 end station

top_end_station = df['end station name'].value_counts()[:10]



fig,ax=plt.subplots(figsize=(20,8))

ax.bar(x=top_end_station.index, height=top_end_station.values, color='#edde68', width=0.5)



#adding value above each bar:Annotation

for p in ax.patches:

    an = ax.annotate(format(p.get_height(), '.2f'), 

                   (p.get_x() + p.get_width() / 2., p.get_height()), 

                   ha = 'center',

                   va = 'center', 

                   xytext = (0, 10), 

                   textcoords = 'offset points')

    an.set_size(20)

ax.set_title("Top 10 end locations in NY",fontsize=30)

ax.set_xlabel("Station name",fontsize=20)



#rotating the x tick labels to 45 degrees

ax.set_xticklabels(top_end_station.index, rotation = 45, ha="right")

ax.set_ylabel("Count",fontsize=20)

#tweeking x and y tick labels 

ax.tick_params(labelsize=15)

plt.show()