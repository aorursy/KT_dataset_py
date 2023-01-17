import plotly.offline as py

py.init_notebook_mode(connected=False)

import plotly_express as px         

import pandas as pd

import matplotlib.pyplot as plt
# The state for running the analysis

STATE = 'AC'



# Associated region for doing comparison

REGION_STATES = ['RO', 'AM', 'RR', 'PA', 'AP', 'AC']
# Load data

filepath = '/kaggle/input/brazilian-bird-observation-metadata-from-wikiaves/wikiaves_metadata-2019-08-11.feather'

data = (pd.read_feather(filepath)

          .assign(registry_date=lambda df: pd.to_datetime(df['registry_date']))

          .assign(location_name=lambda df: pd.Categorical(df['location_name']))

       )
# This block serves for creating an mapping for

# all municipalities names and associated states wih the location id



# An clever trick

location_data = (data.loc[:, ["location_id", "location_name"]]

                    .drop_duplicates(subset=['location_id']))

keys = location_data.location_id.tolist()

values = location_data.location_name.tolist()

location_map = dict(zip(keys, values))



# Generate mappings

state_map = {}

for loc_id, loc_name in location_map.items():

    estado = loc_name.split("/")[-1]

    state_map[loc_id] = estado

    

# Create columns with the mappings

data.loc[:, "location_state"] = data.location_id.map(state_map)

data.loc[:, "home_location_name"] = data.home_location_id.map(location_map)

data.loc[:, "home_location_state"] = data.home_location_id.map(state_map)



# Generate tourist and local definitions

data.loc[:, 'tourist'] = (data.home_location_state != data.location_state)

data.loc[:, 'local'] = (data.home_location_state == data.location_state)



# State and region views

state_data = data.where(lambda df: df['location_state'] == STATE).dropna()

region_data = data.where(lambda df: df.location_state.isin(REGION_STATES)).dropna()
plt.title("Monthly registry fraction on {}".format(STATE))

state_data.groupby(state_data.registry_date.dt.month).tourist.mean().plot(label='tourist')

state_data.groupby(state_data.registry_date.dt.month).local.mean().plot(label='local')

plt.legend()

plt.show()


px.bar(a, x='registry_date', y='tourist')

px.bar
plt.title("Monthly registry count on {}".format(STATE))

state_data.groupby(state_data.registry_date.dt.month).tourist.sum().plot(label='tourist')

state_data.groupby(state_data.registry_date.dt.month).local.sum().plot(label='local')

plt.legend()

plt.show()
plt.title("Yearly registry count on {}".format(STATE))

state_data.groupby(state_data.registry_date.dt.year).tourist.sum().plot(label='tourist')

state_data.groupby(state_data.registry_date.dt.year).local.sum().plot(label='local')

plt.legend()

plt.show()
# Table indicating the registry count and registry from tourists fraction



(state_data.groupby("location_name")

           .tourist

           .agg(['count', 'mean'])

           .sort_values(by='mean', ascending=False)

           .where(lambda df: df['count'] > 5).dropna())
plt.title("Tourist registry fraction on Brazil")

state_data.groupby(state_data.registry_date.dt.year).tourist.mean().plot(label='{}'.format(STATE))

region_data.groupby(data.registry_date.dt.year).tourist.mean().plot(label='region')

data.groupby(data.registry_date.dt.year).tourist.mean().plot(label='Brazil')

plt.legend()

plt.show()
plt.title("Nation-wide tourist registry share for {}".format(STATE))

a = state_data.groupby(state_data.registry_date.dt.year).tourist.sum()

b = region_data.groupby(region_data.registry_date.dt.year).tourist.sum()

c = data.groupby(data.registry_date.dt.year).tourist.sum()

(a / c).plot(label='Nation-wide share')

#(a / b).plot(label='Region-wide share')

plt.legend()

plt.show()
plt.title("Count of distinct registered species on {}".format(STATE))



(state_data[state_data.tourist == 1].resample("3m", on='registry_date')

                                    .species_id

                                    .nunique()

                                    .plot(label='tourist'))



(state_data[state_data.local == 1].resample("3m", on='registry_date')

                                  .species_id

                                  .nunique()

                                  .plot(label='local'))



plt.legend()

plt.show()
species_per_municipality = (state_data.groupby("location_name")

                                      .species_id

                                      .agg(['count', 'nunique'])

                                      .sort_values(by='nunique', ascending=False)

                                      .where(lambda df: df['count'] > 5).dropna())



unique_authors_per_municipality = (state_data.groupby("location_name")

                                             .author_id

                                             .nunique())



species_per_municipality.loc[:, "species/registries"] = (species_per_municipality['nunique'] / species_per_municipality['count'])

species_per_municipality.loc[:, 'species/birdwatchers'] = (species_per_municipality['nunique'] / unique_authors_per_municipality)



species_per_municipality