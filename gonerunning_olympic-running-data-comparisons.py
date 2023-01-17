import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import csv



%matplotlib inline



import time

import datetime
v = open('../input/results.csv', 'r')

r = csv.reader(v)

next(r)

df = pd.read_csv('../input/results.csv', names = 

    ['Gender',

     'Event',

     'Location',

     'Year',

     'Medal',

     'Name',

     'Nationality',

     'Result',

     'Wind'])
df.head()

df["Wind"].fillna(0, inplace=True)
df["Event"].unique()
df['Event'] = df['Event'].str.replace('\sMen|\sWomen', '')
def distance_map(row):

    try:

        return int(row)

    except ValueError:

        return np.nan

    

df['Distance'] = df['Event'].str.replace("M$", "").apply(lambda row: 42195.0 if row == "Marathon" else distance_map(row))
def time_map(row):

    x = datetime.time()

    if np.isnan(row['Distance']):

        return np.nan

    else:

        try:

            x = datetime.datetime.strptime(row['Result'], "%H:%M:%S")

            return datetime.timedelta(hours=x.hour, minutes=x.minute, seconds=x.second).total_seconds()

        except ValueError:

            pass

        try:

            x = datetime.datetime.strptime(row['Result'], "%H:%M:%S.%f")

            return datetime.timedelta(hours=x.hour, minutes=x.minute, seconds=x.second, microseconds = x.microsecond).total_seconds()

        except ValueError:

            pass

        try:

            x = datetime.datetime.strptime(row['Result'], "%Hh%M:%S")

            return datetime.timedelta(hours=x.hour, minutes=x.minute, seconds=x.second).total_seconds()

        except ValueError:

            pass

        try:

            x = datetime.datetime.strptime(row['Result'], "%H-%M:%S")

            return datetime.timedelta(hours=x.hour, minutes=x.minute, seconds=x.second).total_seconds()

        except ValueError:

            pass

        try:

            x = datetime.datetime.strptime(row['Result'], "%H-%M:%S.%f")

            return datetime.timedelta(hours=x.hour, minutes=x.minute, seconds=x.second, microseconds = x.microsecond).total_seconds()

        except ValueError:

            pass

        try:

            x = datetime.datetime.strptime(row['Result'], "%M:%S.%f")

            return datetime.timedelta(hours=x.hour, minutes=x.minute, seconds=x.second, microseconds = x.microsecond).total_seconds()

        except ValueError:

            pass

        try:

            x = datetime.datetime.strptime(row['Result'], "%S.%f")

            return datetime.timedelta(hours=x.hour, minutes=x.minute, seconds=x.second, microseconds = x.microsecond).total_seconds()

        except ValueError:

            pass      

    

df['Seconds'] = df.apply(lambda row: time_map(row), axis=1)

df['Pace'] = df['Seconds']/(df['Distance']/1000)
df['Year'] = df['Year'][1:].apply(int)

g = sns.lmplot(x = "Year", y = "Pace", col = "Event", 

               hue = "Medal", data = df.dropna(), col_wrap = 3)
g = sns.lmplot(x = "Year", y = "Pace", col = "Event", 

               hue = "Gender", data = df.dropna(), col_wrap = 3)
g = sns.regplot(x = "Distance", y = "Pace",

                data = df.dropna(), logx = True)