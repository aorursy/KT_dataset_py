import pandas as pd

df = pd.read_csv('../input/train.csv')

df.head(5)

print(df.shape)

df = df.sample(frac=0.01)

print(df.shape)

df['pickup_datetime'].head(5)
from time import strptime

for index, row in df.iterrows():

    hour = strptime(row['pickup_datetime'], "%Y-%m-%d %H:%M:%S").tm_hour

    if 12 > hour >= 4:

        df.loc[index, "pickup_datetime"] = 'Morning'

    elif 20 > hour >= 12:

        df.loc[index, "pickup_datetime"] = 'Afternoon'

    else:

        df.loc[index, "pickup_datetime"] = 'Night'



df['pickup_datetime'].head(5)
import seaborn as sns

import matplotlib.pyplot as plt

barTime = sns.factorplot(x ='pickup_datetime',y ='trip_duration',data=df,kind='bar')

barTime.set_titles("Duration vs Time of Day")

plt.show(barTime)
latMedian = df['pickup_latitude'].median()

lonMedian = df['pickup_longitude'].median()

print(latMedian,lonMedian)

for index, row in df.iterrows():

    lon = row['pickup_longitude']

    lat = row['pickup_latitude']

    if latMedian < lat and lonMedian < lon:

        df.loc[index, 'pickup_loc_bucket'] = 'NorthEast'

    elif latMedian < lat and lonMedian > lon:

        df.loc[index,'pickup_loc_bucket'] = 'NorthWest'

    elif latMedian > lat and lonMedian < lon:

        df.loc[index,'pickup_loc_bucket'] = 'SouthEast'

    elif latMedian > lat and lonMedian > lon:

        df.loc[index,'pickup_loc_bucket'] = 'SouthWest'

df['pickup_loc_bucket'].head(5)
barLoc = sns.factorplot(x ='pickup_loc_bucket',y ='trip_duration',data=df,kind='bar')

barLoc.set_titles("Duration vs Pickup Location")

plt.show(barLoc)