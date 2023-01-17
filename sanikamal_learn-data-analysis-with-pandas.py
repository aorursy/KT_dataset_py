import pandas as pd

import numpy as np


# data={"day":[0,1,2,3,4,5,6],

#      "sleep":[9,7,6,5,8,7,9],

#      "work":[6,7,9,6,7,8,8]}



# df=pd.DataFrame(data)

# print(df)

death=pd.read_csv('/kaggle/input/uncover/UNCOVER_v4/UNCOVER/USAFacts/confirmed-covid-19-deaths-in-us-by-state-and-county.csv')

death.head()
death.tail()
columns = death.columns

index = death.index

# data = death.to_numpy()

data = death.values
columns
index
data
type(columns)
# columns.to_numpy()

columns.values
type(index)
# index.to_numpy()

index.values
type(data)
death.dtypes
death.dtypes.value_counts()
death.info()
# death['state_name']

death.state_name
death.loc[:,'state_fips']
death.iloc[:,3]
s_attr_methods = set(dir(pd.Series))

print(len(s_attr_methods))

df_attr_methods = set(dir(pd.DataFrame))

print(len(df_attr_methods))

print(len(s_attr_methods & df_attr_methods))
state_name = death['state_name']

deaths_s = death['deaths']

print(state_name.dtype)

print(deaths_s.dtype)
state_name.head()
state_name.sample(n=5,random_state=1100)
state_name.value_counts()
deaths_s.value_counts()
print(state_name.size)

print(state_name.shape)

print(len(state_name))

print(state_name.count())

print(state_name.unique())
print(deaths_s.min())

print(deaths_s.max())

print(deaths_s.mean())

print(deaths_s.median())

print(deaths_s.std())
deaths_s.describe()
print(deaths_s.quantile(0.2))

print(deaths_s.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
death['lat'].isna()
lat_filled = death['lat'].fillna(0)

print(lat_filled.count())
lat_droped = death['lat'].dropna()

print(lat_droped.count())
death['lat'].hasnans
death['lat'].notna()
deaths_s+6
deaths_s-10
deaths_s ** 2
deaths_s > 10
deaths_s != 0
deaths_s.add(6)