import numpy as np

import pandas as pd

import seaborn as sns
df = pd.read_csv('../input/mtl_trajet_points.csv', delimiter='\t')

print('There are {} trips.'.format(len(df.id_trip.unique()))) # There are 68777 trips.
df.drop_duplicates(subset=['id_trip'])['mode'].value_counts() # Trips per mode
# Points per trip; <4 already dropped from dataset

sns.distplot(df['id_trip'].value_counts().value_counts(), bins=20)
df['id_trip'].value_counts().value_counts()