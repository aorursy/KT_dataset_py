import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
dfs = []

for dirname, _, filenames in os.walk('/kaggle/input/airline-2019'):
    for filename in filenames:
        path = os.path.join(dirname, filename)
        dfs.append(pd.read_csv(path))

df = pd.concat(dfs, ignore_index=True)
print('Shape of data: ', df.shape)
df.info(null_counts=True)
df.select_dtypes(exclude=['O'])
df.select_dtypes(include=['O'])
re = r'\s*,\s*[a-zA-Z]*\s*$'
# Clean origin cities
df['ORIGIN_CITY_NAME'] = df['ORIGIN_CITY_NAME'].str.replace(re, '')

# Check
(df['ORIGIN_CITY_NAME'].str.contains(',') == False).sum() == df['ORIGIN_CITY_NAME'].index.size
df['ORIGIN_CITY_NAME']
# Clean destination cities
df['DEST_CITY_NAME'] = df['DEST_CITY_NAME'].str.replace(re, '')

# Check
(df['DEST_CITY_NAME'].str.contains(',') == False).sum() == df['DEST_CITY_NAME'].index.size
df['DEST_CITY_NAME']
df['Unnamed: 25'].value_counts(dropna=False)
(df['Unnamed: 25'].isna()).sum() == df['Unnamed: 25'].index.size
df.drop('Unnamed: 25', axis=1, inplace=True)
df['CANCELLED'].value_counts(dropna=False)
df['CANCELLATION_CODE'].value_counts(dropna=False)
# Split CANCELLATION_CODE into two groups by CANCELLED
df_cancelled_0 = df[df['CANCELLED'] == 0]['CANCELLATION_CODE']
df_cancelled_1 = df[df['CANCELLED'] == 1]['CANCELLATION_CODE']
df_cancelled_0.value_counts(dropna=False)
df_cancelled_1.value_counts(dropna=False)
df.drop('CANCELLED', axis=1, inplace=True)
df['TOTAL_DELAY'] = df['CARRIER_DELAY'] + df['WEATHER_DELAY'] + df['NAS_DELAY'] + df['SECURITY_DELAY'] + df['LATE_AIRCRAFT_DELAY']

df_delays = df[['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']]
ax = df_delays.sum().plot.pie(title='Delays', figsize=(8, 8))
ax.set_xlabel('')
ax.set_ylabel('');
df_airlines = df.groupby('OP_CARRIER_AIRLINE_ID')['TOTAL_DELAY'].aggregate(np.sum).reset_index().sort_values('TOTAL_DELAY', ascending=False)

# Find out the flight operators which correspond to maximum delay in general
plt.figure(figsize=(15, 8))
ax = sns.barplot(x='OP_CARRIER_AIRLINE_ID', y='TOTAL_DELAY', data=df_airlines)
ax.set_xlabel('Airline', fontsize=16)
ax.set_ylabel('Total delay', fontsize=16)
plt.show();
df_airports = df.groupby('ORIGIN')['TOTAL_DELAY'].aggregate(np.sum).reset_index().sort_values('TOTAL_DELAY', ascending=False)

# Find out the airports which correspond to maximum delay in general
plt.figure(figsize=(15, 70))
ax = sns.barplot(y='ORIGIN', x='TOTAL_DELAY', data=df_airports)
ax.set_ylabel('Airport', fontsize=16)
ax.set_xlabel('Total delay', fontsize=16)
plt.show();
df[df['ORIGIN'] == 'ORD'][['ORIGIN', 'ORIGIN_CITY_NAME', 'ORIGIN_STATE_NM']].head()
# Rename columns
# df.rename(columns={'ORIGIN_STATE_NM': 'ORIGIN_STATE_NAME'}, inplace=True)
# df.rename(columns={'DEST_STATE_NM': 'DEST_STATE_NAME'}, inplace=True)