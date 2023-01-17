import numpy as np

import pandas as pd
import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='darkgrid')

%matplotlib inline
df = pd.read_csv('/kaggle/input/police-pedestrian-stops-and-vehicle-stops/police_pedestrian_stops_and_vehicle_stops.csv')
# Take a glance at what we're working with

df.head()
# I like to see the columns in a list

df.columns
# No null values, fortunately

df.isnull().sum()
# Looking deeper at this column

df.CALL_DISPOSITION.value_counts()
# Percentages of some of the more common call dispositions

df.CALL_DISPOSITION.value_counts(normalize=True).head(20)
# Let's breakdown the datetime into a few interesting pieces

df.TIME_PHONEPICKUP = pd.to_datetime(df.TIME_PHONEPICKUP)
df['HOUR'] = df.TIME_PHONEPICKUP.apply(lambda x: x.hour)

df['DAY_OF_WEEK'] = df.TIME_PHONEPICKUP.apply(lambda x: x.weekday())

df['MONTH'] = df.TIME_PHONEPICKUP.apply(lambda x: x.month)

df['YEAR'] = df.TIME_PHONEPICKUP.apply(lambda x: x.year)
# Convert day of week numbers to something we understand

df['DAY_OF_WEEK'] = df['DAY_OF_WEEK'].map({0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'})
# 2010 has far fewer stops than the other years, so let's look deeper

df.YEAR.value_counts()
# Turns out we only have data for December 2010, which means we should probably drop 2010 data, lest the monthly figures become skewed

df[df.YEAR==2010]['MONTH'].value_counts()
# Double-check we are dropping the right stuff!

len(df) - len(df[df.YEAR!=2010])

df = df[df.YEAR!=2010]
hour_vc = df['HOUR'].value_counts().sort_index().to_frame().rename(columns={'HOUR':'COUNT'})

day_of_week_vc = df['DAY_OF_WEEK'].value_counts().sort_index().to_frame().rename(columns={'DAY_OF_WEEK':'COUNT'})

month_vc = df['MONTH'].value_counts().sort_index().to_frame().rename(columns={'MONTH':'COUNT'})

year_vc = df['YEAR'].value_counts().sort_index().to_frame().rename(columns={'YEAR':'COUNT'})
# Plot some timing breakdowns

# Unsurprisingly, we see a significant rise in stops between 10pm and midnight

plt.figure(figsize=(16,6))

ax = sns.barplot(data=hour_vc, x=hour_vc.index, y='COUNT')

ax.set(xlabel='Hour of Day', ylabel='Number of Stops')

ax.set_title('Hourly Breakdown')
# We see an uptick in stops later in the week and on the weekend, possibly due to more police officers on the streets during these days

plt.figure(figsize=(16,6))

day_of_week_vc.index = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

ax = sns.barplot(data=day_of_week_vc, x=day_of_week_vc.index, y='COUNT')

ax.set_ylabel('Number of Stops')

ax.set_title('Day of Week Breakdown')

ax.set_xticklabels(day_of_week_vc.index, rotation=45)
# Hard to make too much of this monthly breakdown

plt.figure(figsize=(16,6))

month_vc.index = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

ax = sns.barplot(data=month_vc, x=month_vc.index, y='COUNT')

ax.set_ylabel('Number of Stops')

ax.set_title('Monthly Breakdown')

ax.set_xticklabels(month_vc.index, rotation=45)
# Stops were relatively steady through 2015, perhaps a policy change let to the substantial rise in 2016/2017

plt.figure(figsize=(16,6))

ax = sns.barplot(data=year_vc, x=year_vc.index, y='COUNT')

ax.set(xlabel='Year', ylabel='Number of Stops')

ax.set_title('Yearly Breakdown')
df.TIME_PHONEPICKUP.max()
# Our last data point is from September 8, 2019. We can extrapolate to predict approximately how many stops there will be in all of 2019.

last_day_of_year = df.TIME_PHONEPICKUP.max().timetuple().tm_yday

perc_through_year = last_day_of_year/365

stops_so_far_2019 = len(df[df.YEAR==2019])

estimated_stops_in_2019 = stops_so_far_2019/perc_through_year

print(estimated_stops_in_2019)
# Since the call dispositions are not completely standardized, we will want to parse a bit to find all the arrests

df[df.CALL_DISPOSITION.apply(lambda x: 'Arrest' in x)].CALL_DISPOSITION.value_counts()
# Let's make a new column returning True if an arrest was made, and false otherwise

df['ARREST_MADE'] = df.CALL_DISPOSITION.str.contains('Arrest')
# What percentage of stops had an arrest made

len(df.ARREST_MADE[df.ARREST_MADE==True])/len(df)

# Or more simply, since True evaluates to 1 and False to 0

df.ARREST_MADE.mean()*100
# Plot the trues and falses as a barplot

plt.figure(figsize=(16,6))

arrest_vc = df['ARREST_MADE'].value_counts(normalize=True).sort_index().to_frame().rename(columns={'ARREST_MADE':'COUNT'})

avc = arrest_vc.reset_index().rename(columns={'index': 'Arrested', 'COUNT': 'Percentage'})

sns.barplot(data=avc, x='Arrested', y='Percentage', palette='Spectral')
# We can plot a random sample of the dataframe with its coordinates and a PROBLEM hue

# This shows us a pretty good rough map of Denver, and where subject stops are more common than vehicle stops

# There are too many neighborhoods for this to work well with a NEIGHBORHOOD_NAME hue, but give it a try, it looks pretty

sns.relplot(data=df.sample(10000), x='GEO_LON', y='GEO_LAT', s=15, alpha=.7, hue='PROBLEM', height=7)
# We can figure out which neighborhoods had the highest rates of subject stops as opposed to vehicle stops

nn_df = df.groupby('NEIGHBORHOOD_NAME').PROBLEM.value_counts(normalize=True).to_frame()

nn_df.rename(columns={'PROBLEM':'Percentage'}, inplace=True)

nn_df.reset_index(inplace=True)

nn_df[nn_df['PROBLEM']=='Subject Stop'].sort_values('Percentage', ascending=False).head()
# We can do the same to find out the chances of being arrested during a stop in a particular neighborhood

nn_df2 = df.groupby('NEIGHBORHOOD_NAME').ARREST_MADE.value_counts(normalize=True).to_frame()

nn_df2.rename(columns={'ARREST_MADE':'Percentage'}, inplace=True)

nn_df2.reset_index(inplace=True)

nn_df2 = nn_df2[nn_df2['ARREST_MADE']==True].sort_values('Percentage', ascending=False)

nn_df2.sort_index(inplace=True)

nn_df2.head()
# By multiplying the number of stops in a neighborhood by percent chance of being arrested, we may be able to detect some of the dodgier bits of Denver

(df.NEIGHBORHOOD_NAME.value_counts().sort_index()*nn_df2['Percentage'].tolist()).to_frame().sort_values('NEIGHBORHOOD_NAME', ascending=False).head(10)