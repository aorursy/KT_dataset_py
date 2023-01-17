# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")



print('Shape of train_df: ', train_df.shape)

train_df.head(10)
train_df['vendor_id'].value_counts()
import math

def Haversine(x):

    lat1 = x['pickup_latitude']

    lat2 = x['dropoff_latitude']

    lon1 = x['pickup_longitude']

    lon2 = x['dropoff_longitude']

    R = 6371000. #meters

    conversion=57.29575

    phi1 = lat1/conversion

    phi2 = lat2/conversion

    deltaPhi = phi1-phi2

    deltaLamb=(lon2-lon1)/conversion

    a=math.sin(deltaPhi/2.)*math.sin(deltaPhi/2.)+math.cos(phi1)*math.cos(phi2)*math.sin(deltaLamb/2.)*math.sin(deltaLamb/2.)

    c=2.*math.atan(math.sqrt(a)*math.sqrt(1.-a))

    return R*c/1000. #kilometers
train_df['Geo_distance'] = train_df.apply(Haversine, axis=1)
train_df.head()
y = np.log(train_df['Geo_distance'] , dtype='float64')

x = np.log(train_df['trip_duration'], dtype='float64')



plt.scatter(x, y, alpha=0.5)

plt.title('Scatter plot distance vs trip duration')

plt.xlabel('Log of trip duration [seconds]')

plt.ylabel('Geodesic distance [kilometers]')

plt.show()
plt.scatter(train_df['trip_duration']/3600., train_df['Geo_distance'], alpha=0.5)

plt.title('Scatter plot distance vs trip duration')

plt.xlabel('trip duration [hours]')

plt.ylabel('Geodesic distance [kilometers]')

plt.show()
train_df[(train_df['trip_duration']> 400.*3600.)]
train_df[(train_df['Geo_distance']> 200.)]
#just drop four entries with long trip durations and eleven with wrong coordinates (these sets do not overlap)



plt.scatter(train_df[(train_df['trip_duration']< 400.*3600.) & (train_df['Geo_distance']< 200.)].trip_duration/3600., 

            train_df[(train_df['trip_duration']< 400.*3600.) & (train_df['Geo_distance']< 200.)].Geo_distance, alpha=0.5)

plt.title('Scatter plot distance vs trip duration')

plt.xlabel('trip duration [hours]')

plt.ylabel('Geodesic distance [kilometers]')

plt.show()
#just drop four entries with long trip durations and eleven with wrong coordinates (these sets do not overlap)



plt.scatter(train_df[(train_df['trip_duration']< 400.*3600.) & (train_df['Geo_distance']< 200.)&(train_df['vendor_id']==1)].trip_duration/3600., 

            train_df[(train_df['trip_duration']< 400.*3600.) & (train_df['Geo_distance']< 200.)&(train_df['vendor_id']==1)].Geo_distance, alpha=0.5)

plt.title('Scatter plot distance vs trip duration VENDOR_ID = 1 ')

plt.xlabel('trip duration [hours]')

plt.ylabel('Geodesic distance [kilometers]')

plt.show()
#just drop four entries with long trip durations and eleven with wrong coordinates (these sets do not overlap)



plt.scatter(train_df[(train_df['trip_duration']< 400.*3600.) & (train_df['Geo_distance']< 200.)&(train_df['vendor_id']==2)].trip_duration/3600., 

            train_df[(train_df['trip_duration']< 400.*3600.) & (train_df['Geo_distance']< 200.)&(train_df['vendor_id']==2)].Geo_distance, alpha=0.5)

plt.title('Scatter plot distance vs trip duration VENDOR_ID = 2 ')

plt.xlabel('trip duration [hours]')

plt.ylabel('Geodesic distance [kilometers]')

plt.show()