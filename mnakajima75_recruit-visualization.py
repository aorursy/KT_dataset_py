import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
air_reserve = pd.read_csv('../input/air_reserve.csv',parse_dates=['visit_datetime','reserve_datetime'])

hpg_reserve = pd.read_csv('../input/hpg_reserve.csv',parse_dates=['visit_datetime','reserve_datetime'])

air_store_info = pd.read_csv('../input/air_store_info.csv')

hpg_store_info = pd.read_csv('../input/hpg_store_info.csv')

store_relation = pd.read_csv('../input/store_id_relation.csv')

date_info = pd.read_csv('../input/date_info.csv',parse_dates=['calendar_date'])

air_visit = pd.read_csv('../input/air_visit_data.csv',parse_dates=['visit_date'])

sample_submission = pd.read_csv('../input/sample_submission.csv')
air_combine = pd.merge(air_reserve, air_store_info, on='air_store_id', how='outer')

air_combine.head(3)
bar = air_combine.groupby(['air_genre_name'],as_index=False).count().sort_values(by='air_store_id',ascending=False)

sns.barplot(y=bar['air_genre_name'], x=bar['air_store_id'], orient='h')

plt.rcParams.update({'font.size': 8})

plt.show()
air_combine[air_combine.air_genre_name == "Japanese food"]['reserve_visitors'].hist(alpha=0.5, bins=50, figsize=(12,6), label="Japanese food", log=True)

air_combine[air_combine.air_genre_name == "Italian/French"]['reserve_visitors'].hist(alpha=0.5, bins=50, figsize=(12,6), label="Italian/French")

air_combine[air_combine.air_genre_name == "Izakaya"]['reserve_visitors'].hist(alpha=0.5, bins=50, figsize=(12,6), label="Izakaya")

air_combine[air_combine.air_genre_name == "Dining bar"]['reserve_visitors'].hist(alpha=0.5, bins=50, figsize=(12,6), label="Dining bar")

air_combine[air_combine.air_genre_name == "Yakiniku/Korean food"]['reserve_visitors'].hist(alpha=0.5, bins=50, figsize=(12,6), label="Yakiniku/Korean food")

air_combine[air_combine.air_genre_name == "Western food"]['reserve_visitors'].hist(alpha=0.5, bins=50, figsize=(12,6), label="Western food")

air_combine[air_combine.air_genre_name == "Cafe/Sweets"]['reserve_visitors'].hist(alpha=0.5, bins=50, figsize=(12,6), label="Cafe/Sweets")

plt.legend(fontsize=16)
print("Japanese food / mean: ", air_combine[air_combine.air_genre_name == "Japanese food"]['reserve_visitors'].mean())

print("Italian/French / mean: ", air_combine[air_combine.air_genre_name == "Italian/French"]['reserve_visitors'].mean())

print("Izakaya / mean: ", air_combine[air_combine.air_genre_name == "Izakaya"]['reserve_visitors'].mean())

print("Dining bar / mean: ", air_combine[air_combine.air_genre_name == "Dining bar"]['reserve_visitors'].mean())

print("Yakiniku/Korean food / mean: ", air_combine[air_combine.air_genre_name == "Yakiniku/Korean food"]['reserve_visitors'].mean())

print("Western food / mean: ", air_combine[air_combine.air_genre_name == "Western food"]['reserve_visitors'].mean())

print("Cafe/Sweets / mean: ", air_combine[air_combine.air_genre_name == "Cafe/Sweets"]['reserve_visitors'].mean())
print("Japanese food / std: ", air_combine[air_combine.air_genre_name == "Japanese food"]['reserve_visitors'].std())

print("Italian/French / std: ", air_combine[air_combine.air_genre_name == "Italian/French"]['reserve_visitors'].std())

print("Izakaya / std: ", air_combine[air_combine.air_genre_name == "Izakaya"]['reserve_visitors'].std())

print("Dining bar / std: ", air_combine[air_combine.air_genre_name == "Dining bar"]['reserve_visitors'].std())

print("Yakiniku/Korean food / std: ", air_combine[air_combine.air_genre_name == "Yakiniku/Korean food"]['reserve_visitors'].std())

print("Western food / std: ", air_combine[air_combine.air_genre_name == "Western food"]['reserve_visitors'].std())

print("Cafe/Sweets / std: ", air_combine[air_combine.air_genre_name == "Cafe/Sweets"]['reserve_visitors'].std())
from datetime import datetime as dt
air_combine["visit_dow"] = air_combine['visit_datetime'].dt.dayofweek

air_combine["reserve_dow"] = air_combine['reserve_datetime'].dt.dayofweek
air_combine[air_combine.visit_dow == 1][air_combine.air_genre_name == "Dining bar"]['reserve_visitors'].hist(alpha=0.5, bins=50, figsize=(12,6), label="Mon", log=True)

air_combine[air_combine.visit_dow == 2][air_combine.air_genre_name == "Dining bar"]['reserve_visitors'].hist(alpha=0.5, bins=50, figsize=(12,6), label="Tue", log=True)

air_combine[air_combine.visit_dow == 3][air_combine.air_genre_name == "Dining bar"]['reserve_visitors'].hist(alpha=0.5, bins=50, figsize=(12,6), label="Wed", log=True)

air_combine[air_combine.visit_dow == 4][air_combine.air_genre_name == "Dining bar"]['reserve_visitors'].hist(alpha=0.5, bins=50, figsize=(12,6), label="Thu", log=True)

air_combine[air_combine.visit_dow == 5][air_combine.air_genre_name == "Dining bar"]['reserve_visitors'].hist(alpha=0.5, bins=50, figsize=(12,6), label="Fri", log=True)

air_combine[air_combine.visit_dow == 6][air_combine.air_genre_name == "Dining bar"]['reserve_visitors'].hist(alpha=0.5, bins=50, figsize=(12,6), label="Sat", log=True)

air_combine[air_combine.visit_dow == 7][air_combine.air_genre_name == "Dining bar"]['reserve_visitors'].hist(alpha=0.5, bins=50, figsize=(12,6), label="Sun", log=True)

plt.legend(fontsize=16)
air_combine[air_combine.visit_dow == 1][air_combine.air_genre_name == "Western food"]['reserve_visitors'].hist(alpha=0.5, bins=50, figsize=(12,6), label="Mon", log=True)

air_combine[air_combine.visit_dow == 2][air_combine.air_genre_name == "Western food"]['reserve_visitors'].hist(alpha=0.5, bins=50, figsize=(12,6), label="Tue", log=True)

air_combine[air_combine.visit_dow == 3][air_combine.air_genre_name == "Western food"]['reserve_visitors'].hist(alpha=0.5, bins=50, figsize=(12,6), label="Wed", log=True)

air_combine[air_combine.visit_dow == 4][air_combine.air_genre_name == "Western food"]['reserve_visitors'].hist(alpha=0.5, bins=50, figsize=(12,6), label="Thu", log=True)

air_combine[air_combine.visit_dow == 5][air_combine.air_genre_name == "Western food"]['reserve_visitors'].hist(alpha=0.5, bins=50, figsize=(12,6), label="Fri", log=True)

air_combine[air_combine.visit_dow == 6][air_combine.air_genre_name == "Western food"]['reserve_visitors'].hist(alpha=0.5, bins=50, figsize=(12,6), label="Sat", log=True)

air_combine[air_combine.visit_dow == 7][air_combine.air_genre_name == "Western food"]['reserve_visitors'].hist(alpha=0.5, bins=50, figsize=(12,6), label="Sun", log=True)

plt.legend(fontsize=16)