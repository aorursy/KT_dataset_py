import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

% matplotlib inline
# Load and process store information

df_air_store = pd.read_csv('../input/air_store_info.csv')

df_hpg_store = pd.read_csv('../input/hpg_store_info.csv')



df_air_store = df_air_store.rename(columns={

    'air_store_id': 'store_id',

    'air_genre_name': 'genre_name',

    'air_area_name': 'area_name'

})



df_hpg_store = df_hpg_store.rename(columns={

    'hpg_store_id': 'store_id',

    'hpg_genre_name': 'genre_name',

    'hpg_area_name': 'area_name'

})



# Add dataset information

df_air_store['type'] = 'air'

df_hpg_store['type'] = 'hpg'



# Combine the datasets

df_store = pd.concat([df_air_store, df_hpg_store], axis=0)
# Load and process reservation information

df_air_reserve = pd.read_csv('../input/air_reserve.csv')

df_hpg_reserve = pd.read_csv('../input/hpg_reserve.csv')



df_air_reserve = df_air_reserve.rename(columns={'air_store_id': 'store_id'})

df_hpg_reserve = df_hpg_reserve.rename(columns={'hpg_store_id': 'store_id'})



df_air_reserve['type'] = 'air'

df_hpg_reserve['type'] = 'hpg'



df_reserve = pd.concat([df_air_reserve, df_hpg_reserve], axis=0)

df_reserve['visit_datetime'] = pd.to_datetime(df_reserve['visit_datetime'])

df_reserve['reserve_datetime'] = pd.to_datetime(df_reserve['reserve_datetime'])
df_joined = pd.merge(df_reserve, df_store, left_on='store_id', right_on='store_id')
df_joined['lag_days'] = (df_joined['visit_datetime'] - df_joined['reserve_datetime']).apply(lambda x: x.total_seconds() / 3600. / 24.)
df_joined['lag_days'][df_joined['lag_days'] < 30].hist(normed=True)

plt.title('Reservation lag')

plt.xlabel('No. days')

plt.ylabel('Frequency')

plt.show()
df_joined['reserve_visitors'] = df_joined['reserve_visitors'].astype(int)

visitors_lag = df_joined.groupby('reserve_visitors').apply(lambda x: x['lag_days'].mean())

visitors_lag.keys = ['visitors', 'lag_days']

visitors_lag = visitors_lag.to_frame('Lag days')

visitors_lag[visitors_lag.index < 60].plot()
from sklearn.neural_network import MLPRegressor



samples = df_joined.sample(25000)



X = samples['reserve_visitors'].values.reshape(-1, 1) / 100.

y = samples['lag_days'] / 30.



model = MLPRegressor(hidden_layer_sizes=(100, 100,), activation='relu')

model.fit(X, y)



X = np.arange(1, 100).reshape(-1, 1)

y = model.predict(X / 100.) * 30.
fig, ax = plt.subplots(1, 1)

ax.plot(X, y)



visitors_lag.plot(ax=ax)

plt.show()
fig, ax = plt.subplots(1, 1)

ax.plot(X[:40], y[:40])



visitors_lag[visitors_lag.index < 40].plot(ax=ax)

plt.show()
df_joined['reserve_visitors'] = df_joined['reserve_visitors'].astype(int)

visitors_lag = df_joined.groupby('reserve_visitors').apply(lambda x: x['lag_days'].std())

visitors_lag.keys = ['visitors', 'lag_days']

visitors_lag = visitors_lag.to_frame('Lag days')

visitors_lag[visitors_lag.index < 60].plot()