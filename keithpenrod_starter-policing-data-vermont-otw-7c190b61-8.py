from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

print(os.listdir('../input'))
nRowsRead = 10000 # specify 'None' if want to read whole file

# VT-clean.csv has 283285 rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('../input/VT-clean.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'VT-clean.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
df1.info()
df1.describe()
df1.loc[df1.driver_age_raw.notna() & df1.driver_age.isna()]
df1.loc[df1.driver_race_raw.notna() & df1.driver_race.isna()]
df1.loc[df1.driver_race_raw.notna() & df1.driver_race.isna()].driver_race_raw.value_counts()
df2 = df1.loc[df1.driver_race.notna() & df1.driver_gender.notna()]

race_gender = pd.crosstab(df2.driver_race, df2.driver_gender)

race_gender
race_gender.plot.bar()
race_gender.drop('White',axis=0).plot.bar()
race_gender.M/race_gender.F
pd.crosstab(df1.driver_race, df1.search_conducted, normalize='index')
pd.crosstab(df1.driver_race, df1.is_arrested, normalize='index')
pd.crosstab(df1.search_conducted, df1.is_arrested, normalize='index')
searched = df1.loc[df1.search_conducted]

pd.crosstab(searched.driver_race, searched.is_arrested)
df1.state.value_counts()
df_fixed = df1.drop('state', axis=1)
df1.county_name.nunique()
df_fixed['county_name'] = df_fixed.county_name.astype('category')
df_fixed.police_department.value_counts()
df_fixed['police_department'] = df_fixed.police_department.astype('category')
df_fixed['driver_gender'] = df_fixed.driver_gender.astype('category')

df_fixed['driver_race'] = df_fixed.driver_race.astype('category')
df_fixed.violation_raw.value_counts()
df_fixed['violation'] = df_fixed.violation.astype('category')

df_fixed['violation_raw'] = df_fixed.violation_raw.astype('category')
df_fixed.search_type_raw.value_counts()
df_fixed.search_type.value_counts()
df_fixed.drop(['driver_age_raw','search_type_raw'], axis=1, inplace=True)
df_fixed['driver_race_raw'] = df_fixed.driver_race_raw.astype('category')

df_fixed['search_type'] = df_fixed.search_type.astype('category')
df_fixed.stop_outcome.value_counts()
df_fixed['stop_outcome'] = df_fixed.stop_outcome.astype('category')
df_fixed.info()
df_fixed['stop_datetime'] = pd.to_datetime(df_fixed.stop_date + ' ' + df_fixed.stop_time)
df_fixed.drop(['stop_date','stop_time'], axis=1, inplace=True)

df_fixed.set_index('stop_datetime', inplace=True)
df_fixed['2010-07-01']
df_fixed['2010-07-01 10:00':'2010-07-01 11:00']