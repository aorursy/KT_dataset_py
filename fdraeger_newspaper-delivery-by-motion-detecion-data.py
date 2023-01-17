# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# read data & set the timestamp dtype

contacts = pd.DataFrame.from_csv("../input/contacts_2017-5-26.csv")

contacts.timestamp = contacts.timestamp.apply(pd.to_datetime)

print (contacts.contact.unique(),"\n")



# _Main_Door only, ignore the rest

contacts = contacts[contacts.contact=="_Main_Door"]

print (contacts.head(5))
# get the motion data and set the timestamp dtype

motion = pd.DataFrame.from_csv("../input/motion_2017-5-26.csv")

motion.timestamp = motion.timestamp.apply(pd.to_datetime)

print (motion.head(5))
# getting an overview of motion data



md = motion[['tod','isDetected']]

# look only at motion data

md = md[md.isDetected==True]

# function to print motion data in time slices

def showMotion(motionFrom = "00:00",motionUntil = "23:59"):

    tmd[(tmd.index>=motionFrom)&(tmd.index < motionUntil)].plot(kind="line",style="b.-"\

    ,grid=True,title="Motions by Hour of Day from %s to %s"%(motionFrom,motionUntil)\

    ,figsize=(14,6),legend=False)



#

# this counts True as 1 and False as 0

#

tmd = md.pivot_table(index="tod",aggfunc=sum)



showMotion(motionFrom = "00:00",motionUntil = "23:59")

showMotion(motionFrom = "00:00",motionUntil = "08:00")
import datetime as dt



def findEventsNearby( ts, tdelta = 10, dataset = contacts):

    #

    # for a given timestamp ts, check if there is 

    # a record in dataset in +/- tdelta seconds

    #

    delta = dt.timedelta(seconds=tdelta)

    minTime = ts - delta

    maxTime = ts + delta

    #print minTime, " < ", ts, " < ", maxTime

    tmplist = dataset[(dataset.timestamp >= minTime) \

    & (dataset.timestamp <= maxTime) ]

    #& (dataset.timestamp.dt.dayofweek != 0)]        # do not count sundays

    return tmplist



# timespan where to look for open front door: +/- vicinitySecs

vicinitySecs = 15



# empty result Series

soleMotion=None

soleMotion = pd.DataFrame(data=[], dtype=pd.datetime)



# iterate over all motions

for ts in motion.timestamp:

    # Except Sunday

    if ts.weekday() != 6:

        l = findEventsNearby(ts,tdelta=vicinitySecs, dataset=contacts)

        if l.size == 0:

            # nothing found, seems to be a sole motion

            soleMotion = soleMotion.append([ts], ignore_index=True)



soleMotion = soleMotion.rename_axis({0: "timestamp"}, axis="columns")



print ("Found %d motions where the front door did not open within +- %d seconds." %(soleMotion.size,vicinitySecs))

tmp=None # just to be sure

tmp = pd.DataFrame(data=[],index=soleMotion.timestamp)

# create a count column to use for aggregation

tmp['cnt'] = 1

tmp = tmp.resample('5T').sum()  # resample/aggregate data in x minute chunks

tmp['ts'] = tmp.index

tmp['tod'] = tmp.ts.dt.time

# Pivot Table

ttt = tmp.pivot_table(index='tod',aggfunc=sum)



# ttt now has aggregated motion counts by time of day
motionFrom = dt.time(0,0)

motionUntil= dt.time(10,0)

ttt[(ttt.index>=motionFrom)&(ttt.index < motionUntil)].plot(kind="bar",style="b"\

    ,grid=True,title="Motions by Hour of Day from %s to %s"%(motionFrom,motionUntil)\

    ,figsize=(14,6),legend=False)
motionFrom = dt.time(3,0)

motionUntil= dt.time(7,0)

ttt[(ttt.index>=motionFrom)&(ttt.index < motionUntil)].plot(kind="bar",style="b.-"\

    ,grid=True,title="Motions by Hour of Day from %s to %s"%(motionFrom,motionUntil)\

    ,figsize=(14,6),legend=False)