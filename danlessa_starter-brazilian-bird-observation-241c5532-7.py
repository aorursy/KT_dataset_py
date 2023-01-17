import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as st

data_path = '/kaggle/input/wikiaves_metadata-2019-08-11.feather'
data = pd.read_feather(data_path)



data = (data.assign(away_flag=lambda df: df['location_id'] != df['home_location_id'])

            .assign(changed_away=lambda df: (df['away_flag'].astype(int).diff() != 0))

            .assign(location_group=lambda df: df.groupby('author_id')['changed_away'].cumsum())

       )
travel_groups = data.groupby(['author_id', 'location_group'])

travel_date_groups =  travel_groups['registry_date']

travel_periods = (travel_date_groups.max() - travel_date_groups.min()).dt.total_seconds() / (60 * 60 * 24) + 1
travel_periods.hist(log=True, range=(0, 60), bins=60)

plt.show()
author_groups = data.groupby("author_id")

species_per_author = author_groups["species_id"].nunique()
species_per_author.hist(log=True, bins=200)

plt.show()
species_per_author_groups = data.groupby(['author_id', 'species_id'])

first_species_observation = species_per_author_groups['registry_date'].min()
aa = first_species_observation.sort_values().groupby("author_id")

aa_min = first_species_observation.min()

bb = aa.apply(lambda x: x - aa_min).dt.total_seconds() / (60 * 60 * 24)
cc = bb.groupby('author_id').cumcount()

dd = pd.concat([bb, cc], axis=1)

dd.columns = ["days", "species_count"]
dd.groupby('days').max().plot()
plt.figure(figsize=(15, 10))

plt.xlabel("Quantidade de dias")

plt.ylabel("Número de espécies distintas observadas")

plt.plot(dd.days / 30, dd.species_count, '.', alpha=0.005, markersize=15)

plt.xticks(np.arange(0, 150, 3))

plt.xlim((60, 96))

plt.ylim((0, 800))

plt.grid()

plt.show()
plt.figure(figsize=(12, 9))

y = data.resample("1m", on="registry_date")['species_id'].nunique()

plt.plot(y, 'x-')

plt.grid()

plt.title("Espécies distintas observadas no Wikiaves mensalmente")

plt.ylabel("Número de espécies distintas")

plt.xlabel("Data")

plt.show()
plt.figure(figsize=(12, 9))

y = data.resample("1m", on="registry_date")['species_id'].count()

plt.plot(y, 'x-')

plt.grid()

plt.title("Número de registros mensais no Wikiaves")

plt.ylabel("Número de espécies distintas")

plt.xlabel("Data")

plt.show()
sp_data = data.where(lambda df: df['location_id'] == 3550308).dropna()



suiriri_data = data.where(lambda df: df['species_id'] == 11338).dropna()
y = data.groupby(data.registry_date.dt.dayofweek)['registry_date'].count()





plt.figure(figsize=(12, 9))

plt.plot(y, 'x-')

plt.grid()

plt.title("Número de registros mensais no Wikiaves")

plt.ylabel("Número de espécies distintas")

plt.xlabel("Data")

plt.show()