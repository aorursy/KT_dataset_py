



import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Input data files are available in the "../input/" directory.

path = 'D:/BACKUP/Kaggle/New York City Taxi/Data/'

train_df = pd.read_csv('../input/train.csv')



#--- Let's peek into the data

print (train_df.head())
#--- Check if there are any Nan values ---

train_df.isnull().values.any()
#--- First see the range of values present ---

print (max(train_df['trip_duration'].values))

print (min(train_df['trip_duration'].values))
#train_df = train_df.assign(trip_duration_mins = lambda x: int(x.trip_duration/60))

train_df['trip_duration_mins'] = train_df.apply(lambda row: row['trip_duration'] / 60, axis=1)

print (train_df.head())
print (max(train_df['trip_duration_mins'].values))

print (min(train_df['trip_duration_mins'].values))
train_df.plot(x=train_df.index, y='trip_duration_mins')
long_rides = []

short_rides = []

count_short = 0

count_long = 0



for i in range(0, train_df.shape[0]):

    if train_df['trip_duration_mins'][i] > 20000:

        long_rides.append(train_df['id'][i])

        count_long+=1

    elif train_df['trip_duration_mins'][i] < 1:

        short_rides.append(train_df['id'][i])

        count_short+=1

        

print ("These are {} the ids that had long rides.".format(long_rides))

print (count_long)

print ("These are {} the ids that had short rides.".format(short_rides))

print (count_short)

#--- First let us count the number of unique passenger counts in the data set ---

print ("These are {} unique passenger counts.".format(train_df['passenger_count'].nunique()))



#--- Well what are those counts ? ---

print (train_df['passenger_count'].unique())
#--- Now let us plot them against the index and see their distribution.



pd.value_counts(train_df['passenger_count']).plot.bar()
#--- But we want to see the actual count ---

#train_df.groupby(train_df['passenger_count']).count()

train_df['passenger_count'].value_counts()
print (train_df['trip_duration_mins'].corr(train_df['passenger_count']))
#--- Seeing the rows of data having longest cab rides ---

train_df[train_df.id.isin(long_rides)]