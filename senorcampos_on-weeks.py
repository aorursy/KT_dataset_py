# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3 as sql

import csv

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


#df_train['time'].describe()
time = df_train['time']
time = time % (24*60)#*60#*60*10

n, bins, patches = plt.hist(time, 50)
plt.title('What is Time?')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
#So, 1 of 3 things is happening: Nothing happens cyclically with time; time isn't based on hours, minutes, seconds, or sub-seconds; or they've given us data that aggregates to the same use for each hour.
#Option 3 sounds most promising, so let's dive into that
#df_train['place_id'].value_counts().head(10) #get the top places to breakout time
offset=0 # This can be adjusted if we figure out what time midnight is
#Let's take a look at how each individual place breaks down with time
time = df_train[df_train['place_id']==8772469670]['time']

timeToTest=24*60#*60#*60*10

time = (time+offset) % timeToTest

n, bins, patches = plt.hist(time, 50)
plt.title('What is Time?')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
time = df_train[df_train['place_id']==1623394281]['time']
# check for cycles over the course of a day (top) and week (bottom)
def plotDayAndWeek(time,offset,clumpHours):
    fig, axs = plt.subplots(2,1);
    for j in range(0,2):
        timeToTest=24*60*(j*6+1)
        timeC = (time+offset) % timeToTest
        n, bins, patches = axs[j].hist(timeC/60., 24 / clumpHours * (j*6+1));
        plt.xlabel('Time (Hours)');
        plt.ylabel('Frequency');
        fig.show();
time = df_train[df_train['place_id']==8772469670]['time']
clumpHours = 1  # this could be used to reduce the number of histogram bins
# later on we'll want to know what are the likely places at a given time
# in that case, a "given time" probably doesn't need minute-by-minute resolution
# in fact, too much resolution in time would be noisier 
plotDayAndWeek(time,offset,clumpHours)
# looks like this place is open for the first 12 hours the cycle, so cycle prob starts around 6am?
# third weekly peak is the biggest... maybe taco tuesday? which would put sunday 6am on left
time = df_train[df_train['place_id']==1623394281]['time']
plotDayAndWeek(time,offset,clumpHours)
# looks like a nighttime / weekend place, where weekend can mean early morning
# nighttime might even mean 6 in the morning
# could be a place that everyone goes to on a weekday night and party all night
# are there people on this site that do that?!  We might not have the right expertise.
time = df_train[df_train['place_id']==1308450003]['time']
plotDayAndWeek(time,offset,clumpHours)
time = df_train[df_train['place_id']==4823777529]['time']
plotDayAndWeek(time,offset,clumpHours)
# I wouldn't want to work at this place
#Strong case for this dataset being in minutes.
# weekly cycles add supporting evidence
# still don't know what the two big dips in the middle of time range...

#Let's see how much time this data has been collected for
print('Time since start of data collection: ' + str(round(df_train['time'].max()/(24*60*365.25),2)) + ' Years.')
# the long time period also makes sense given a steady overall rise of checkins over time
# check for cycles over the course of a year - but 7-day week should still dominate
def plotYear(time,offset):
    timeToTest=24*60*365
    timeC = (time+offset) % timeToTest
    n, bins, patches = plt.hist(timeC/(60.*24.*7.), 52*7);
    plt.xlabel('Time (Weeks)');
    plt.ylabel('Frequency');
    plt.show();
time = df_train[df_train['place_id']==8772469670]['time']
plotYear(time,offset)
time = df_train[df_train['place_id']==1623394281]['time']
plotYear(time,offset)
time = df_train[df_train['place_id']==1308450003]['time']
plotYear(time,offset)
# have to check if this place actually closed after 35ish weeks into the first year
# if so, a popular place would account for 0 checkins in the test set.
# specifically, did checkins come back in the new year, or were they done after 35 weeks?
time = df_train[df_train['place_id']==1308450003]['time']
timeC = time /(60.*24.*7.)
plt.plot(np.sort(timeC))
# the very few-and-far between checkins for this place after 35ish weeks (y-axis) means two things
# 1) the place closed after 30ish weeks - exclude it from test set predicitons
# 2) some checkins to a place can be considered erroneous
#    in the sense that the person didn't actually go to that place

# late-stage tweaking may include a "closed for business" detector"
# other places might have few checkins, but also be new, so should be scaled higher than the raw count

time = df_train[df_train['place_id']==4823777529]['time']
plotYear(time,offset)
