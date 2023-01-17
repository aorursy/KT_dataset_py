import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv('../input/bikes.csv')

df.head(3)
df.columns
df.status.value_counts()
df.banking.value_counts()
df = df[df.status != 'CLOSED']

df.head()
df.bonus.value_counts()
print(df.name.value_counts())

len(df.groupby('name', as_index=False).groups)
import matplotlib.pyplot as plt

import numpy as np



ax = df_byName = df.groupby('name').mean().sort_values(['bike_stands'], ascending=False)['bike_stands'].plot(kind='bar')
df_byName = df.groupby('name').mean().sort_values(['available_bike_stands'], ascending=False)['available_bike_stands']

ax = df_byName.plot(kind='bar')

df_shortage = df[df.available_bike_stands < 1]

df_shortage_groupped = df_shortage.groupby('name').count().sort_values(['available_bike_stands'], ascending=False)['available_bike_stands']

print(df_shortage_groupped)

ax = df_shortage_groupped.plot(kind='bar')
stations_with_shortage = list(df_shortage_groupped.index)



df_noshortage = df[~df.name.isin(stations_with_shortage)]



df_noshortage_groupped = df_noshortage.groupby('name').mean().sort_values(['available_bike_stands'], ascending=False)['available_bike_stands']

print(df_noshortage_groupped)

ax = df_noshortage_groupped.plot(kind='bar')