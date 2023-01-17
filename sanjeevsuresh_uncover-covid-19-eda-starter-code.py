# Libraries Needed

import pandas as pd

import os

from glob import glob

from tqdm.notebook import tqdm

from collections import namedtuple

from pathlib import Path

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go





# Some display options to easily eyeball the dataframes and their contents

pd.set_option('display.max_columns', 200)

pd.set_option('display.max_colwidth', -1)

pd.set_option('display.max_rows', 1000)
Dataset = namedtuple('Dataset', ['df', 'source'])

datasets = {}



# Reading all csv files as Datasets

for dirname, _, filenames in os.walk('/kaggle/input'):

    for csv_path in glob(os.path.join(dirname, "*.csv")):

        file_name = os.path.basename(csv_path)

        name = os.path.splitext(file_name)[0]

        source = os.path.basename(Path(csv_path).parent)

        datasets[name] = Dataset(pd.read_csv(csv_path, low_memory=False), source)



print("Read a total of {} datasets".format(len(datasets)))
for name, dataset in sorted(datasets.items(), key=lambda x: x[1].df.shape[0], reverse=True):

    print("{name} ({source}) - {shape}".format(name=name, source=dataset.source, shape=dataset.df.shape))
cases = datasets['johns-hopkins-covid-19-daily-dashboard-cases-over-time'].df

# Finding total cases per counrtry and adding mortality percent 

countries_cases = cases.groupby('country_region', as_index=False).max().assign(mortality_percent=lambda x: round(x.deaths*100.0/x.confirmed, 1))



fig = make_subplots(rows=4, cols=1, subplot_titles=('Top Confirmed cases', 'Top Deaths reported', 'Top Mortality percent', 'Mortality percent (among countries having the most confirmed cases)'))



# selecting the 15 countries which had the most confirmed

top_confirmed_countries = countries_cases.sort_values('confirmed').tail(15)

fig.add_trace(go.Bar(x=top_confirmed_countries["confirmed"], y=top_confirmed_countries["country_region"],

                     text=top_confirmed_countries["confirmed"], textposition='outside', orientation='h', marker_color='rgb(255,69,0)'), row=1, col=1)



# selecting the 15 countries which had the most deaths

top_deaths_countries = countries_cases.sort_values('deaths').tail(15)

fig.add_trace(go.Bar(x=top_deaths_countries["deaths"], y=top_deaths_countries["country_region"],

                     text=top_deaths_countries["deaths"], textposition='outside', orientation='h', marker_color='rgb(220,20,60)'), row=2, col=1)



# selecting the 15 countries which had the most mortality percent(that is deaths per 100 cases)

top_mortality_percent_countries = countries_cases.sort_values('mortality_percent').tail(15)

fig.add_trace(go.Bar(x=top_mortality_percent_countries["mortality_percent"], y=top_mortality_percent_countries["country_region"],

                     text=top_mortality_percent_countries["mortality_percent"], textposition='outside', orientation='h', marker_color='rgb(240,128,128)'), row=3, col=1)



# selecting the 15 countries which had the most confirmed, looking at mortality percent among that

mortality_percent_top_confirmed_countries = countries_cases.sort_values(by='confirmed').tail(15).sort_values(by='mortality_percent')

fig.add_trace(go.Bar(x=mortality_percent_top_confirmed_countries["mortality_percent"], y=mortality_percent_top_confirmed_countries["country_region"],

                     text=mortality_percent_top_confirmed_countries["mortality_percent"], textposition='outside', orientation='h', marker_color='rgb(205,92,92)'), row=4, col=1)



fig.update_layout(showlegend = False, height=2000)

fig.show()
top_countries = list(top_confirmed_countries['country_region'])

cases_by_top_countries = cases[cases['country_region'].isin(top_countries)]

fig = px.line(cases_by_top_countries, x="last_update", y="confirmed", color='country_region', title='Confirmed Cases By Country')

fig.update_layout(xaxis_title="Date", yaxis_title='Confirmed Count')

fig.show()
fig = px.line(cases_by_top_countries, x="last_update", y="deaths", color='country_region', title='Deaths By Country')

fig.update_layout(xaxis_title="Date", yaxis_title='Death Count')

fig.show()
world_bank_country_map = {"United States": "US", "Iran, Islamic Rep.": "Iran", "Korea, Rep.": "Korea, South"}

global_population = datasets['global-population'].df

# Handling US & Iran which is represented different in this dataset

global_population['country'].replace(world_bank_country_map, inplace=True)

global_population = global_population[global_population['country'].isin(top_countries)][['country', 'year_2018']]
covid_stats_with_population_df = pd.merge(global_population, top_confirmed_countries, left_on='country', right_on='country_region', how='inner')

fig = px.scatter(x=covid_stats_with_population_df['year_2018'], y=covid_stats_with_population_df['mortality_percent'], color=covid_stats_with_population_df['country'])

fig.update_layout(title="Effect of total population on mortality", xaxis_title="Population Count", yaxis_title='Mortality Percent')

fig.show()

# Filtering for only the relevant countries for this metric

hospital_beds = datasets['hospital-beds-per-1-000-people'].df.replace(world_bank_country_map).query('country_name in @top_countries')

# Trying to find the latest year for each country with non null data. Since the columns are years, first we are normalising the data for easier handling.

hospital_beds_normalised = hospital_beds.melt(id_vars=["country_name", "country_code", "indicator_name", "indicator_code"], var_name="Year", value_name="value")

hospital_beds_normalised['Year'] = pd.to_numeric(hospital_beds_normalised['Year'])

hospital_beds_normalised = hospital_beds_normalised[hospital_beds_normalised['value'].notna()].groupby('country_name', as_index=False).apply(lambda x: x.nlargest(1, 'Year'))



covid_stats_with_hospital_beds_df = pd.merge(covid_stats_with_population_df, hospital_beds_normalised, left_on='country_region', right_on='country_name', how='inner')

fig = px.scatter(x=covid_stats_with_hospital_beds_df['value'], y=covid_stats_with_hospital_beds_df['mortality_percent'], color=covid_stats_with_hospital_beds_df['country'])

fig.update_layout(title="Effect of hospital beds on mortality", xaxis_title="Hospital Beds Per 1000 people", yaxis_title='Mortality Percent')

fig.show()