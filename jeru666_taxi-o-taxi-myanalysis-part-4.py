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
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])

train_df['dropoff_datetime'] = pd.to_datetime(train_df['dropoff_datetime'])



train_df['pickup_month'] = train_df.pickup_datetime.dt.month.astype(np.uint8)

train_df['pickup_day'] = train_df.pickup_datetime.dt.weekday.astype(np.uint8)

train_df['pickup_hour'] = train_df.pickup_datetime.dt.hour.astype(np.uint8)



train_df['dropoff_month'] = train_df.dropoff_datetime.dt.month.astype(np.uint8)

train_df['dropoff_day'] = train_df.dropoff_datetime.dt.weekday.astype(np.uint8)

train_df['dropoff_hour'] = train_df.dropoff_datetime.dt.hour.astype(np.uint8)

print (train_df.head())

from math import radians, cos, sin, asin, sqrt   #--- for the mathematical operations involved in the function ---



def haversine(lon1, lat1, lon2, lat2):

    """

    Calculate the distance between two points 

    on the earth (specified in decimal degrees)

    """

    # convert decimal degrees to radians 

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 

    dlon = lon2 - lon1 

    dlat = lat2 - lat1 

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2

    c = 2 * asin(sqrt(a)) 

    km = 6367 * c

    return km



train_df['Displacement (km)'] = train_df.apply(lambda x: haversine(x['pickup_longitude'], x['pickup_latitude'], x['dropoff_longitude'], x['dropoff_latitude']), axis=1)



print (train_df.head())
#--- First let us count the number of unique vendor_ids in the data set ---

print ("These are {} unique vendor ids.".format(train_df['vendor_id'].nunique()))



#--- Well what are those counts ? ---

print (train_df['vendor_id'].unique())
pd.value_counts(train_df['vendor_id']).plot.bar()
print(train_df['vendor_id'].value_counts())
s = train_df['trip_duration'].groupby(train_df['vendor_id']).sum()

print (s)
s = train_df['trip_duration'].groupby(train_df['vendor_id']).mean()

print (s)
print('Mean pickup_longitude')

print(train_df['pickup_longitude'].groupby(train_df['vendor_id']).mean())

print(' ')

print('Mean pickup_latitude')

print(train_df['pickup_latitude'].groupby(train_df['vendor_id']).mean())
print('Mean pickup_month')

print(train_df['pickup_month'].groupby(train_df['vendor_id']).mean())

print(' ')

print('Mean pickup_day')

print(train_df['pickup_day'].groupby(train_df['vendor_id']).mean())

print(' ')

print('Mean pickup_hour')

print(train_df['pickup_hour'].groupby(train_df['vendor_id']).mean())

#--- Now let us count the number of unique store_and_fwd_flags in the data set ---

print ("These are {} unique store_and_fwd_flags.".format(train_df['store_and_fwd_flag'].nunique()))



#--- Well what are those counts ? ---

print (train_df['store_and_fwd_flag'].unique())
#--- Let us plot them against the index and see their distribution.



pd.value_counts(train_df['store_and_fwd_flag']).plot.bar()
print(train_df['store_and_fwd_flag'].value_counts())


train_df['store_and_fwd_flag'] = train_df['store_and_fwd_flag'].astype('category')

train_df['store_and_fwd_flag'] = train_df['store_and_fwd_flag'].cat.codes



#---Now let us count them again ---

print (train_df['store_and_fwd_flag'].unique())



print(train_df.head())
print('Mean trip duration for each flag')

print(train_df['trip_duration'].groupby(train_df['store_and_fwd_flag']).mean())

print (' ')

print('Mean Displacement in km for each flag')

print(train_df['Displacement (km)'].groupby(train_df['store_and_fwd_flag']).mean())
print (train_df['store_and_fwd_flag'].value_counts())
print('Mean pickup_month')

print(train_df['pickup_month'].groupby(train_df['store_and_fwd_flag']).mean())

print(' ')

print('Mean pickup_day')

print(train_df['pickup_day'].groupby(train_df['store_and_fwd_flag']).mean())

print(' ')

print('Mean pickup_hour')

print(train_df['pickup_hour'].groupby(train_df['store_and_fwd_flag']).mean())