# native libs

import os



# Data handling

import pandas as pd

import numpy as np



import plotly.graph_objects as go

import chart_studio.plotly as py



%load_ext autoreload

%autoreload 2

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/emission data.csv', delimiter=',')

df.head()
df[df['1751'] != 0]
# only select the UK and the world

only_uk_world = df[(df["Country"] == "United Kingdom") | (df["Country"] == "World")]

# get the difference, drop the Country column since it contains strings

diff_uk_world = only_uk_world.drop(columns ="Country").diff(axis=0)
import plotly.express as px



fig = go.Figure(data=go.Scatter(x=df.columns, y=df.iloc[1]))

fig.show()
import pycountry

pycountry.countries.search_fuzzy(df.Country.iloc[0])[0].alpha_3
some_countries = ["England", "HerpaDerpaland", "Engla"]



def do_fuzzy_search(country):

    try:

        result = pycountry.countries.search_fuzzy(country)

        return result[0].alpha_3

    except:

        return np.nan



for country in some_countries:

    print(do_fuzzy_search(country))
df['country_code'] = df["Country"].apply(lambda country: do_fuzzy_search(country))

df.head()
plot_df = df.dropna()

fig = px.choropleth(plot_df, locations="country_code",

                    color="2017",

                    hover_name="Country",

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.show()
missing_countries = ["Congo", "Democratic Republic of Congo", "Niger", "South Korea"]

correct_codes = {"Congo": "COD", "Democratic Republic of Congo": "COG", "Niger": "NER", "South Korea": "KOR"}

df[df["Country"].isin(missing_countries)]
def update_wrong_country_codes(row):

    if row['Country'] in correct_codes.keys():

        row['country_code'] = correct_codes[row['Country']]

    return row



df = df.apply(lambda x: update_wrong_country_codes(x), axis=1)
plot_df = df.dropna()

fig = px.choropleth(plot_df, locations="country_code",

                    color="2017",

                    hover_name="Country",

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.show()
import plotly



# constants

first_year = 1900

last_year = 2017

number_of_steps = int(2017 - 1900)



# data is a list that will have one element for now, the first element is the value from the first year column we are interested in.

data = [dict(type='choropleth',

             locations = plot_df['country_code'].astype(str),

             z=plot_df[str(first_year)].astype(float))]



# next, we copy the data from the first cell, append it to the data list and set the data to the value contained in the year column of the dataframe.

for i in range(number_of_steps):

    data.append(data[0].copy())

    index = str(first_year + i + 1)

    data[-1]['z'] = plot_df[index]
# for each entry in data, we add one step to the slider bar and define a label e.g. Year 1900

steps = []

for i in range(len(data)):

    step = dict(method='restyle',

                args=['visible', [False] * len(data)],

                label='Year {}'.format(i + first_year))

    step['args'][1][i] = True

    steps.append(step)



sliders = [dict(active=number_of_steps,

                pad={"t": 1},

                steps=steps)]    

layout = dict(sliders=sliders)
fig = dict(data=data, 

           layout=layout)

plotly.offline.iplot(fig)