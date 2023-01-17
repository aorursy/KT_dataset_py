# First, we'll import pandas, a data processing and CSV file I/O library

import pandas as pd

import numpy as np

import datetime



# We'll also import seaborn, a Python graphing library

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)



# Remove comments, if you want to see files.

#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Routine to parse dates

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')



# Read data from both files



d=pd.read_csv("../input/trainView.csv",

    header=0,names=['train_id','status','next_station','service','dest','lon',

                    'lat','source','track_change','track','date','timeStamp0',

                    'timeStamp1','seconds'],

    dtype={'train_id':str,'status':str,'next_station':str,'service':str,'dest':str,

    'lon':str,'lat':str,'source':str,'track_change':str,'track':str,'date':str,

    'timeStamp0':datetime.datetime,'timeStamp1':datetime.datetime,'seconds':str}, 

     parse_dates=['timeStamp0','timeStamp1'],date_parser=dateparse)







o=pd.read_csv("../input/otp.csv",

    header=0,names=['train_id','direction','origin','next_station','date','status',

                    'timeStamp'],

    dtype={'train_id':str,'direction':str,'origine':str,'next_station':str,

                           'date':str,'status':str,'timeStamp':datetime.datetime}, 

    parse_dates=['timeStamp'],date_parser=dateparse)



# How many stations, and how many trains

# at each station for the chosen DATE?



DATE = '2016-05-22'

g=o[o['date'] == DATE ].groupby(['next_station']).count().reset_index()[['next_station','train_id']]

g.columns = ['next_station', 'count']

g.head(160)
# Clean up status 

#  1. Fix status to be integers

#  2. Remove None





def fixStatus(x):

   x=x.replace('On Time','0')

   x=x.replace(' min','')

   return int(x)



o['Delay']=o['status'].apply(lambda x: fixStatus(x))

o = o[(o['next_station'] != 'None')]



o[['Delay','status','next_station']].head()
# Maybe Create Categories For Times ... EarlyMorning, MorningRush ..

#



# Divide up times

#d['timeStamp'] = d['timeStamp'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))







def timeOfDay(n):

    if n.time() < datetime.time(6, 30):

        return 'EarlyMorning'

    if n.time() >= datetime.time(6, 30) and n.time() <= datetime.time(9, 30):

        return 'MorningRush'

    if n.time() > datetime.time(9, 30) and n.time() < datetime.time(16, 0):

        return '9to4pm'

    

    if n.time() >= datetime.time(16, 0) and n.time() <= datetime.time(18, 30):

        return 'EveningRush'

    if n.time() > datetime.time(18, 30):

        return 'After630pm'





o['dayCat'] = o['timeStamp'].apply(lambda x: timeOfDay(x))

o['hr'] = o['timeStamp'].apply(lambda x: x.hour)

# weekday - 0 == Monday, >6 is weekend

o['weekday'] = o['timeStamp'].apply(lambda x: x.weekday())





# Let's see the results

o[['timeStamp','dayCat','hr','weekday','next_station']].head()
# Let's take a look at all trains at 'Elkins Park'

# Trains only run North and South.  Does it make

# difference, when looking at delay?



# Take one day

ep=o[(o['next_station']=='Elkins Park') & (o['date']>= '2016-05-22' ) & 

     (o['date']< '2016-05-31')]



# Train is considered late >= 6 min

epd=ep[ep['Delay']>=6]

epd['Delay'].describe()





# Let's graph the Delay by direction, for the late trains

g = sns.FacetGrid(epd, col="direction")

g.map(plt.hist, "Delay");
# This is tricky – some trains may cross their 24 hour boundry

#   (See timeDiff below on how these are ignored)



# You can plug in values here. 



# Let's look at a month of trains between two stations.

STATION_A='Elkins Park'

STATION_B='Suburban Station'

DATE_BEGIN = '2016-05-01'

DATE_END = '2016-05-30'



# The train_id passes through STATION_A and it passes through STATION_B



ab=o[(o.train_id.isin(o[o['next_station']==STATION_A]['train_id'])) & 

     (o.train_id.isin(o[o['next_station']==STATION_B]['train_id']))]







# All train_id's that go through both points

train_id_ab=ab['train_id'].unique()



pointA=o[o.train_id.isin(train_id_ab) & (o['next_station']==STATION_A)][['train_id','timeStamp','next_station','date']]





pointB=o[o.train_id.isin(train_id_ab) & (o['next_station']==STATION_B)][['train_id','timeStamp','next_station','date']]







# Merge pointA and pointB

line=pd.merge(pointA,pointB, left_on=['train_id','date'], right_on=['train_id','date'])





# Routine for getting time differences.

import datetime

def timeDiff(x):

    if x[1] > x[4]:

        c = x[1] - x[4]

    else:

        c = x[4] - x[1]

    if c.total_seconds() >= 25200:  # Train disable/cross 24 hr.  7hrs

        return np.nan  # Ignore these

    return c.total_seconds()







# Date Range

line = line[(line['date'] >= DATE_BEGIN) & (  line['date'] < DATE_END       )]



line['seconds']=line.apply(timeDiff, axis = 1)



# line[line['seconds'].isnull()]  # These are problems

line = line[line['seconds'].notnull()]  # Remove the problems

line['minutes']= line['seconds']/60  # No one thinks in seconds. Convert to min.

line[line['timeStamp_x'] > line['timeStamp_y']].head(7)
# About 1,260 trains? Seems correct for a month

line[line['timeStamp_x'] > line['timeStamp_y']]['minutes'].describe()
# Going the other direction

line[line['timeStamp_x'] < line['timeStamp_y']]['minutes'].describe()
# Graph it

sns.set_style("whitegrid")

ax=sns.distplot(line['minutes'])



ax.set(xlabel='Minutes', 

       ylabel='% Distribution',title=''+str(STATION_A)+' to '+

       str(STATION_B) +' ')



plt.legend();