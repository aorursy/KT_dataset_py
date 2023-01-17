import numpy as np
import pandas as pd
collisions = pd.read_csv('../input/nypd-motor-vehicle-collisions.csv')
collisions.head()
# Distribution of Collisions by Borough
collisions['BOROUGH'].value_counts().plot(kind='bar')
# Look at only Collisions where a person was either killed or injured
(collisions
    .loc[(collisions["NUMBER OF PERSONS INJURED"] > 0) | collisions["NUMBER OF PERSONS KILLED"] > 0]
    .loc[:,'BOROUGH']
    .value_counts()
    .plot(kind='bar')
)
# Deadliest collisions

(collisions
     .iloc[collisions.groupby("BOROUGH")["NUMBER OF PERSONS KILLED"].idxmax()]
     .loc[:,['DATE', 'BOROUGH']]
     .sort_values(by='DATE', ascending=False)
)
# Distribution of Collisions by Date
# Time of day not recorded

collisions['DATE'] = pd.to_datetime(collisions['DATE'], format="%Y-%m-%d")
collisions['DATE'].value_counts().resample('m').sum().plot.line()
# Causes