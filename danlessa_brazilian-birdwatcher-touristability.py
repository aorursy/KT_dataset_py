import pandas as pd

import matplotlib.pyplot as plt

import geopy.distance

import numpy as np

data_filepath = '/kaggle/input/brazilian-bird-observation-metadata-from-wikiaves/wikiaves_metadata-2019-08-11.feather'

cities_filepath = '/kaggle/input/brazilian-cities/BRAZIL_CITIES.csv'



# Birdwatching data

data = (pd.read_feather(data_filepath)

          .assign(registry_date=lambda df: pd.to_datetime(df['registry_date']))

          .assign(location_name=lambda df: pd.Categorical(df['location_name']))

          .assign(scientific_species_name=lambda df: pd.Categorical(df['scientific_species_name']))

          .assign(popular_species_name=lambda df: pd.Categorical(df['popular_species_name']))

          .assign(species_wiki_slug=lambda df: pd.Categorical(df['species_wiki_slug']))

          .assign(city=lambda df: pd.Categorical(df.location_name.apply(lambda row: row.split("/")[0])))

          .assign(state=lambda df: pd.Categorical(df.location_name.apply(lambda row: row.split("/")[1])))

       )



# Data for Brazilian cities

cities_data = (pd.read_csv(cities_filepath, delimiter=';')

                 .assign(location_name=lambda df: df.apply(lambda x:"{}/{}".format(x.CITY, x.STATE).lower(), axis=1))

                 .set_index("location_name")

              )
# Generate an location_id -> location_name map

location_data = (data.loc[:, ["location_id", "location_name"]]

                    .drop_duplicates(subset=['location_id']))

keys = location_data.location_id.tolist()

values = location_data.location_name.tolist()

location_map = dict(zip(keys, values))



data.loc[:, 'location_name'] = data.location_name.str.lower()

data.loc[:, "home_location_name"] = data.home_location_id.map(location_map)
data = data.sort_values(['registry_date', 'author_id', 'registry_id'])

data = (data.assign(away_flag=lambda df: df['location_id'] != df['home_location_id'])

            .assign(changed_away=lambda df: (df['away_flag'].astype(int).diff() != 0))

            .assign(location_group=lambda df: df.groupby('author_id')['changed_away'].cumsum())

       )
travel_groups = data.groupby(['author_id', 'location_group'])

travel_date_groups =  travel_groups['registry_date']

travel_periods = (travel_date_groups.max() - travel_date_groups.min()).dt.total_seconds() / (60 * 60 * 24) + 1

travel_periods.hist(log=True, range=(0, 60), bins=60)

plt.ylabel("Travel count")

plt.xlabel("Travel time in days")

plt.show()
travel_periods.sum() / 365.25
travel_groups = data.groupby(['author_id', 'location_group'])

travel_date_groups =  travel_groups['registry_date']

travel_periods = (travel_date_groups.max() - travel_date_groups.min()).dt.total_seconds() / (60 * 60 * 24) + 1

travel_periods.hist(range=(0, 150), bins=40, normed=True, cumulative=True)

plt.ylabel("Travel count")

plt.xlabel("Travel time in days")

plt.show()
t_index = data.head(100000).groupby("author_id").rolling(2).location_id.apply(lambda x: (x.iloc[1] != x.iloc[0]), raw=False)
t_index.groupby("author_id").mean().hist(log=True, bins=20)
data['location_sparse_group'] = t_index.groupby("author_id").cumsum().droplevel(0)

travel_groups = data.groupby(['author_id', 'location_sparse_group'])

travel_date_groups =  travel_groups['registry_date']

travel_periods = (travel_date_groups.max() - travel_date_groups.min()).dt.total_seconds() / (60 * 60 * 24) + 1

travel_periods.hist(log=True, range=(0, 60), bins=60)

plt.ylabel("Travel count")

plt.xlabel("Travel time in days")

plt.show()
travel_groups = data.groupby(['author_id', 'location_group'])

travel_date_groups =  travel_groups['registry_date']

travel_periods = travel_date_groups.nunique()

travel_periods.hist(log=True, range=(0, 40), bins=40)

plt.ylabel("Travel count")

plt.xlabel("Days with registries outside hometown")

plt.show()
travel_periods.sum() / 365.25
travel_groups = data.groupby(['author_id', 'location_group'])

travel_date_groups =  travel_groups['registry_date']

travel_periods = travel_date_groups.nunique()

travel_periods.hist(range=(0, 20), bins=20, cumulative=True, normed=True)

plt.ylabel("Travel count")

plt.xlabel("Days with registries outside hometown")

plt.show()
travel_groups = data.groupby(['author_id', 'location_sparse_group'])

travel_date_groups =  travel_groups['registry_date']

travel_periods = travel_date_groups.nunique()

travel_periods.hist(log=True, range=(0, 40), bins=40)

plt.ylabel("Travel count")

plt.xlabel("Days with registries outside hometown")

plt.show()
## Cumulative distance per author
lat_lon_map = cities_data.loc[:, ['LAT', 'LONG']].to_dict(orient='index')

lat_lon_df = pd.DataFrame(lat_lon_map).T

located_data = data.join(lat_lon_df, on='location_name').dropna(subset=['LAT', 'LONG'])
locs = located_data.groupby(["author_id", 'registry_date', 'location_id']).registry_id.count()
def strides_2d(a, r, linear=True):

    

    ax = np.zeros(shape=(a.shape[0] + 2*r[0], a.shape[1] + 2*r[1]))

    ax[:] = np.nan

    ax[r[0]:ax.shape[0]-r[0], r[1]:ax.shape[1]-r[1]] = a

    

    shape = a.shape + (1+2*r[0], 1+2*r[1])

    strides = ax.strides + ax.strides

    s = as_strided(ax, shape=shape, strides=strides)

    

    return s.reshape(a.shape + (shape[2]*shape[3],)) if linear else s
strides_2d(cities_data[['LAT', 'LONG']].values, 2)
from geopy.distance import vincenty

def coord_distance(x):

    lat1, lat2 = x.index.values

    lng1, lng2 = x.values

    p1 = (lat1, lng1)

    p2 = (lat2, lng2)

    return vincenty(p1, p2).km



aff = located_data[['LAT', 'LONG']].set_index("LAT")

bff = aff.LONG.rolling(2).apply(coord_distance, raw=False)