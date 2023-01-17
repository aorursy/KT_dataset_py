# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('seaborn-whitegrid')
#As data size is huge we will be dealing with a million rows only
%time
df_train =  pd.read_csv('../input/train.csv', nrows = 2_000_000, parse_dates=["pickup_datetime"])

# list first few rows (datapoints)
df_train.head()
df_train.describe()
df_train = df_train.dropna(how = 'any', axis = 'rows')
df_train.describe()

df_train = df_train.drop(df_train[df_train['fare_amount']<0].index, axis=0)
df_train.shape
df_train['fare_amount'].value_counts().sort_index().plot.line(xlim=(0,1300))
df_train['fare_amount'].describe()
df_train[df_train.fare_amount<100].fare_amount.hist(bins=100, figsize=(14,3))
df_train = df_train[(df_train.fare_amount>=2.5) & (df_train.fare_amount<=100) ]
df_train['fare_amount'].describe()
df_train['passenger_count'].describe()
df_train = df_train[(df_train.passenger_count < 7) & (df_train.passenger_count>0 )]
df_train['passenger_count'].value_counts().sort_index().plot.bar()

def haversine_distance(lat1, long1, lat2, long2):
    data = [df_train]
    for i in data:
        R = 6371  #radius of earth in kilometers
        #R = 3959 #radius of earth in miles
        phi1 = np.radians(i[lat1])
        phi2 = np.radians(i[lat2])
    
        delta_phi = np.radians(i[lat2]-i[lat1])
        delta_lambda = np.radians(i[long2]-i[long1])
    
        #a = sin²((φB - φA)/2) + cos φA . cos φB . sin²((λB - λA)/2)
        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    
        #c = 2 * atan2( √a, √(1−a) )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
        #d = R*c
        d = (R * c) #in kilometers
        i['H_Distance'] = d
    return d
haversine_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')
data = [df_train]
for i in data:
    i['Year'] = i['pickup_datetime'].dt.year
    i['Month'] = i['pickup_datetime'].dt.month
    i['Date'] = i['pickup_datetime'].dt.day
    i['Day of Week'] = i['pickup_datetime'].dt.dayofweek
    i['Hour'] = i['pickup_datetime'].dt.hour
df_train.head()
plt.figure(figsize=(15,7))
plt.hist(df_train['passenger_count'], bins=15)
plt.xlabel('No. of Passengers')
plt.ylabel('Frequency')
plt.figure(figsize=(15,7))
plt.scatter(x=df_train['passenger_count'], y=df_train['fare_amount'], s=1.5)
plt.xlabel('No. of Passengers')
plt.ylabel('Fare')
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")

# Generate a large random dataset
d = pd.DataFrame(data=rs.normal(size=(100, 26)),
                 columns=list(ascii_letters[26:]))

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.figure(figsize=(15,7))
plt.scatter(df_train['Date'], y=df_train['fare_amount'], s=1.4)
plt.xlabel('Date')
plt.ylabel('Fare')
plt.figure(figsize=(15,7))
plt.hist(df_train['Day of Week'], bins=100)
plt.xlabel('Day of Week')
plt.ylabel('Frequency')
plt.figure(figsize=(15,7))
plt.hist(df_train['Hour'], bins=100)
plt.xlabel('Hour')
plt.ylabel('Frequency')
plt.figure(figsize=(15,7))
plt.scatter(x=df_train['Hour'], y=df_train['fare_amount'], s=1.5)
plt.xlabel('Hour')
plt.ylabel('Fare')
