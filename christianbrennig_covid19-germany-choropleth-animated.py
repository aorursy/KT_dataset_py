import json

import datetime

import pandas as pd

import urllib.request

import plotly.express as px
url = '../input/rki-covid19-dataset/data'

rki_data = pd.read_csv(url, usecols=['IdLandkreis', 'Meldedatum', 'AnzahlFall'], 

                       dtype={'IdLandkreis': str}, parse_dates=['Meldedatum'])

rki_data.head()
print('Rows: ', len(rki_data), ' Cases: ', rki_data.AnzahlFall.sum())
berlin_ids = [str(x) for x in range(11001, 11013)]

new_ids = ['11000' for x in range(11001, 11013)]

replace_ids = dict(zip(berlin_ids, new_ids))

rki_data = rki_data.replace(replace_ids)
rki_data = rki_data.groupby(['Meldedatum','IdLandkreis']).sum()

rki_data = rki_data.sort_values(by=['IdLandkreis', 'Meldedatum'])

rki_data = rki_data.reset_index()

rki_data.head()
rki_data.dtypes
print('Rows: ', len(rki_data), ' Cases: ', rki_data.AnzahlFall.sum())
with open('../input/countiespolygonssmaller/counties_polygons_smaller.json') as file:

    counties_polygons = json.load(file)
population_data = pd.read_csv('../input/population-germanycsv/population_germany.csv', sep=';', dtype={'county_id':str})

population_data.head()
population_data.dtypes
print('Rows: ', len(population_data), ' Total: ', population_data.population.sum())
weeks_review = abs(round((rki_data.Meldedatum.min() - rki_data.Meldedatum.max()).days/7)) - 1
def select_periode(weeks=3):

    first_record_date = rki_data.Meldedatum.min()

    most_recent_record_date = rki_data.Meldedatum.max()

    periods = {}

    for week in range(weeks):

        periode = [most_recent_record_date - datetime.timedelta(days=(7+week*7)) , most_recent_record_date - datetime.timedelta(days=(week*7))]

        if periode[0] > first_record_date:

            periods[(weeks-week)] = periode

    return periods

periods = select_periode(weeks_review)
def fill_zero_records(df, date):

    for county_id in population_data.county_id:

        if county_id not in df.IdLandkreis.values:

            df = df.append({'IdLandkreis': county_id, 'AnzahlFall': 0.0, 'week': date} , ignore_index=True) 

    return df
def append_periode_data(periode):

    mask = (rki_data['Meldedatum'] > periode[0]) & (rki_data['Meldedatum'] <= periode[1])

    periode_data = rki_data.loc[mask]

    periode_data = periode_data.groupby(['IdLandkreis']).sum()

    periode_data = periode_data.sort_values(by=['IdLandkreis'])

    periode_data = periode_data.reset_index()

    periode_data['week'] = [periode[1] for n in range(len(periode_data))]

    periode_data = fill_zero_records(periode_data, periode[1])

    return periode_data
plot_data = pd.DataFrame(columns=['IdLandkreis', 'AnzahlFall', 'week'])

for i in range(1, (weeks_review+1)):

    plot_data = plot_data.append(append_periode_data(periods[i]))
plot_data = plot_data.merge(population_data, left_on='IdLandkreis', right_on='county_id')

plot_data['per100000'] = round(100000 * (plot_data.AnzahlFall / plot_data.population), 1)

plot_data = plot_data.astype({'week': 'str'})

plot_data.head()
print('Rows: ', len(plot_data), 'Cases selected: ', plot_data.AnzahlFall.sum())
def plot_choropleth_map(counties_polygons, df, max_range=90):

    fig = px.choropleth_mapbox(df, geojson=counties_polygons, locations="county_id", color="per100000",

                               color_continuous_scale=["green", "yellow", "red", "purple"],

                               featureidkey="properties.krs_code",

                               hover_name="name",

                               range_color=(0, max_range),

                               animation_frame="week",

                               mapbox_style="carto-positron",

                               zoom=4.5, center={"lat": 51.2, "lon": 11.03283},

                               opacity=0.8)

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000

    fig.show("notebook")
plot_choropleth_map(counties_polygons, plot_data)