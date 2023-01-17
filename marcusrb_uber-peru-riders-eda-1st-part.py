# Load and import main packages and libraries

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

import datetime





%matplotlib inline

plt.rcParams['figure.figsize'] = (16.0, 12.0)
# Check main directory

import os

os.getcwd()
# Load the dataset from local directory to new var, uber2010

dataset = "../input/uber_peru_2010.csv"

uber2010 = pd.read_csv(dataset, sep=";")
# Get first 10 results

uber2010.head(n=10)
# Get info about our dataset

uber2010.info()
# keep the original dataset safetly and create new one to manipulate the data

uber2010_bak = uber2010
# Create 4 new variables, dayofweek number and name, month of day and hour of day.

uber2010['start_at'] = pd.to_datetime(uber2010['start_at'], format="%d/%m/%Y %H:%M")

uber2010['DayOfWeekNum'] = uber2010['start_at'].dt.dayofweek

uber2010['DayOfWeek'] = uber2010['start_at'].dt.weekday_name

uber2010['MonthDayNum'] = uber2010['start_at'].dt.day

uber2010['HourOfDay'] = uber2010['start_at'].dt.hour
#converting start_at column to datetime 

#uber2010['start_at'] =  pd.to_datetime(uber2010['start_at'], format='%Y-%m-%d %H:%M:%S')





#extracting the Hour from start_at column and putting it into a new column named 'Hour'

uber2010['Hour'] = uber2010.start_at.apply(lambda x: x.hour)



#extracting the Minute from start_at column and putting it into a new column named 'Minute'

uber2010['Minute'] = uber2010.start_at.apply(lambda x: x.minute)



#extracting the Month from start_at column and putting it into a new column named 'Month'

uber2010['Month'] = uber2010.start_at.apply(lambda x: x.month)



#extracting the Day from start_at column and putting it into a new column named 'Day'

uber2010['Day'] = uber2010.start_at.apply(lambda x: x.day)



#extracting the Weekday from start_at column and putting it into a new column named 'WeekDay'

uber2010['WeekDay'] = uber2010.start_at.apply(lambda x: x.strftime('%A'))

#browse updated dataset

uber2010.iloc[:10,:5]
uber2010.iloc[:10,5:15]
uber2010.iloc[:10,15:28]
uber2010.iloc[:10,28:40]
# Lets we observe stats data for only numerical variables

uber2010.describe()
# Check & drop N/a's - part 1

uber2010.isnull().sum()
# Remove na values

uber2010 = uber2010.dropna()
uber2010.isnull().any()
uber2010.info()
# Let we check how many unique journey or riders we have

print("Uber2010 dataset has {} unique journeys, or riders.".format(uber2010['journey_id'].nunique()))
# Let we check the rider's scoring

meanScore = round(np.mean(uber2010['rider_score']),2)

maxScore = np.count_nonzero(uber2010['rider_score'] == 5)

minScore = np.count_nonzero(uber2010['rider_score'] == 1)

print("Them have an avg score of {} of five...".format(meanScore))

print("{} of the whole riders, scored 5 of 5...".format(maxScore))

print("...and {} scored 1 of 5...".format(minScore))
# Let we check how many unique vehicles we have

print("The dataset has {} unique vehicle...".format(uber2010['taxi_id'].nunique()))
# Let we check how many unique vehicles we have

print("...for {} unique drivers.".format(uber2010['driver_id'].nunique()))
# Let we check the driver's scoring

meanScore = round(np.mean(uber2010['driver_score']),2)

maxScore = np.count_nonzero(uber2010['driver_score'] == 5)

minScore = np.count_nonzero(uber2010['driver_score'] == 1)

print("Them have an avg score of {} of five...".format(meanScore))

print("{} of the whole riders, scored 5 of 5...".format(maxScore))

print("...and {} scored 1 of 5...".format(minScore))
listIcons = str(pd.unique(uber2010['icon']))

print("Uber2010 dataset has {} unique icon and listed...".format(uber2010['icon'].nunique()), listIcons)
listSources = str(pd.unique(uber2010['source']))

print("The main sources are {} and listed...".format(uber2010['source'].nunique()), listSources)
# Check how many unique values we have

uber2010.nunique()
# Let’s create a pivot_table() of the number of riders each driver and taxi picked on each day:

riders_by_taxi = uber2010.pivot_table(index='start_at', columns='taxi_id', values='journey_id', aggfunc='count')

riders_by_taxi.head()
# Lets we visualize total riders for each day of week

uber2010_weekdays = uber2010.pivot_table(index=['DayOfWeekNum','DayOfWeek'],

                                  values='journey_id',

                                  aggfunc='count')

uber2010_weekdays.plot(kind='bar', figsize=(15,6))

plt.ylabel('Total riders')

#plt.set_xticklabels(x_labels)

plt.title('Riders by Week Day');
#group the data by Weekday and hour

summary = uber2010.groupby(['WeekDay', 'Hour'])['start_at'].count()
#reset index

summary = summary.reset_index()

#convert to dataframe

summary = pd.DataFrame(summary)

#browse data

summary.head()
#rename last column

summary=summary.rename(columns = {'start_at':'Counts'})
tableau_color_blind = [(0, 107, 164), (255, 128, 14), (171, 171, 171), (89, 89, 89),

             (95, 158, 209), (200, 82, 0), (137, 137, 137), (163, 200, 236),

             (255, 188, 121), (207, 207, 207)]



for i in range(len(tableau_color_blind)):  

    r, g, b = tableau_color_blind[i]  

    tableau_color_blind[i] = (r / 255., g / 255., b / 255.)
sns.set_style('whitegrid')



## set palette   

current_palette = sns.color_palette(tableau_color_blind)

plt.figure(figsize=(15,6))

ax = sns.pointplot(x="Hour", y="Counts", hue="WeekDay", data=summary, palette = current_palette,)

handles,labels = ax.get_legend_handles_labels()

#reordering legend content

handles = [handles[1], handles[5], handles[6], handles[4], handles[0], handles[2], handles[3]]

labels = [labels[1], labels[5], labels[6], labels[4], labels[0], labels[2], labels[3]]

ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

ax.set_xlabel('Hour', fontsize = 12)

ax.set_ylabel('Count of Uber Pickups', fontsize = 12)

ax.set_title('Hourly Uber Pickups By Day of the Week in Peru (Jan-X 2010)', fontsize=16)

ax.tick_params(labelsize = 8)

ax.legend(handles,labels,loc=0, title="Legend", prop={'size':8})

ax.get_legend().get_title().set_fontsize('8')

plt.show()
# Best idea is to copy a backup of original dataset and it transform

uber2010_new = uber2010_bak
# Remove na values

uber2010_new = uber2010_new.dropna()
uber2010_new['start_at_ptz'] = uber2010_new['start_at'].dt.tz_localize('Etc/GMT+1').dt.tz_convert('America/Lima')
uber2010_new['start_at_ptz'].head()
# Comparison old vs new data timezone

uber2010['start_at'].head()
#extracting the Hour from start_at column and putting it into a new column named 'Hour'

uber2010_new['Hour'] = uber2010_new.start_at_ptz.apply(lambda x: x.hour)



#extracting the Minute from start_at column and putting it into a new column named 'Minute'

uber2010_new['Minute'] = uber2010_new.start_at_ptz.apply(lambda x: x.minute)



#extracting the Month from start_at column and putting it into a new column named 'Month'

uber2010_new['Month'] = uber2010_new.start_at_ptz.apply(lambda x: x.month)



#extracting the Day from start_at column and putting it into a new column named 'Day'

uber2010_new['Day'] = uber2010_new.start_at_ptz.apply(lambda x: x.day)



#extracting the Weekday from start_at column and putting it into a new column named 'WeekDay'

uber2010_new['WeekDay'] = uber2010_new.start_at_ptz.apply(lambda x: x.strftime('%A'))
#group the data by Weekday and hour

summary_new = uber2010_new.groupby(['WeekDay', 'Hour'])['start_at_ptz'].count()
#reset index

summary_new = summary_new.reset_index()

#convert to dataframe

summary_new = pd.DataFrame(summary_new)

#browse data

summary_new.head()
#rename last column

summary_new=summary_new.rename(columns = {'start_at_ptz':'Counts'})
sns.set_style('whitegrid')



## set palette   

current_palette = sns.color_palette(tableau_color_blind)

plt.figure(figsize=(15,6))

ax = sns.pointplot(x="Hour", y="Counts", hue="WeekDay", data=summary_new, palette = current_palette,)

handles,labels = ax.get_legend_handles_labels()

#reordering legend content

handles = [handles[1], handles[5], handles[6], handles[4], handles[0], handles[2], handles[3]]

labels = [labels[1], labels[5], labels[6], labels[4], labels[0], labels[2], labels[3]]

ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

ax.set_xlabel('Hour', fontsize = 12)

ax.set_ylabel('Count of Uer Pickups', fontsize = 12)

ax.set_title('Hourly Uber Pickups By Day of the Week in Peru (Jan-X 2010)', fontsize=16)

ax.tick_params(labelsize = 8)

ax.legend(handles,labels,loc=0, title="Legend", prop={'size':8})

ax.get_legend().get_title().set_fontsize('8')

plt.show()
# Uber aggregate pickups by the hour in Peru

sns.set_style('whitegrid')

plt.figure(figsize=(15,6))

ax = sns.countplot(x="Hour", data=uber2010_new, color="lightsteelblue")

ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

ax.set_xlabel('Hour of the Day', fontsize = 12)

ax.set_ylabel('Count of Uber Pickups', fontsize = 12)

ax.set_title('Uber pickups by the Hour in Peru (Jan-X 2010)', fontsize=16)

ax.tick_params(labelsize = 8)

plt.show()
# Uber pickups by the month

sns.set_style('whitegrid')

plt.figure(figsize=(15,6))

ax = sns.countplot(x="Month", data=uber2010_new, color="lightsteelblue")

ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

ax.set_xlabel('Month', fontsize = 12)

ax.set_ylabel('Count of Uber Pickups', fontsize = 12)

ax.set_title('Uber pickups by the Month in Peru (Jan-X 2010)', fontsize=16)

ax.tick_params(labelsize = 8)

plt.show()
uber_monthdays = uber2010_new.pivot_table(index=['MonthDayNum'],

                                  values='journey_id',

                                  aggfunc='count')

uber_monthdays.plot(kind='bar', figsize=(15,6))

plt.ylabel('Total Riders')

plt.title('Riders by Month Day');
uber_hour = uber2010_new.pivot_table(index=['HourOfDay'],

                                  values=summary,

                                  aggfunc='count')

uber_hour.plot(kind='barh', figsize=(15,6))

plt.ylabel('Total Journeys')

plt.title('Journeys by Hour');
latMin = uber2010_new['start_lat'].min()

latMin
latMax = uber2010_new['start_lat'].max()

latMax
lonMin = uber2010_new['start_lon'].min()

lonMin
lonMax = uber2010_new['start_lon'].max()

lonMax