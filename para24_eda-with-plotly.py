import os, sys

from collections import defaultdict

from urllib.request import urlopen

import json



import numpy as np

import pandas as pd



import plotly.graph_objects as go

from plotly.subplots import make_subplots

from ipywidgets import widgets





# For ipywidgets to work on kaggle

# !pip install -q ipywidgets

# !jupyter nbextension enable --py --sys-prefix widgetsnbextension





# Included changes to make the kernel run as a jupyter notebook on windows without the need to make any changes

kaggle_data_folder = os.path.join('kaggle', 'input') if sys.platform == 'win32' else os.path.join(os.path.sep, 'kaggle', 'input')

file_exts = (".csv", ".geojson")

all_files = defaultdict(list)

for dirname, _, files in os.walk(kaggle_data_folder):

    for file in files:

        name, ext = os.path.splitext(file)

        if ext in file_exts:

            all_files[ext].append(os.path.join(dirname, file))



print(all_files)
df = pd.read_csv(all_files['.csv'][0])

df.columns = ['Category', 'Name', 'LRate_Total_2001', 'LRate_Total_2011',

              'LRate_Rural_2001', 'LRate_Rural_2011', 'LRate_Urban_2001', 'LRate_Urban_2011']



df.head(2)
stubnames1 = ['LRate_Total', 'LRate_Rural','LRate_Urban']

stubnames2 = ['LRate']

df = pd.wide_to_long(df, stubnames1, i="Name", j="Year", sep='_')



# Sort to get back 'INDIA' rows on the top

df.sort_values(by=['Category', 'Name'], inplace=True)



# Reset index

df.reset_index(inplace=True)



# Reorder Columns

temp = df['Category']

df.drop(labels=['Category'], axis=1, inplace = True)

df.insert(0, 'Category', temp)
df.head(4)
df['Rural_pop'] = round((df['LRate_Total'] - df['LRate_Urban']) / (df['LRate_Rural'] - df['LRate_Urban']) * 100, 1)

df['Urban_pop'] = 100 - df['Rural_pop']



df.head(3)
## Read the polygon information of various Indian states from the GeoJSON file

with open(all_files['.geojson'][0], 'r') as fp:

    india = json.load(fp)
geo_df = pd.DataFrame(data=[st['properties']['NAME_1'] for st in india['features']], columns=['State Names in GeoJSON'])

data_df = pd.DataFrame(df.iloc[2:, 1].unique(), columns=['State Names in DATA'])

geo_df.merge(data_df,

             how='outer',

             left_on='State Names in GeoJSON',

             right_on='State Names in DATA')
mapper = {'Jammu & Kashmir': 'Jammu and Kashmir',

          'Odisha': 'Orissa',

          'Uttarakhand': 'Uttaranchal',

          'A & N Islands': 'Andaman and Nicobar',

          'D & N Haveli': 'Dadra and Nagar Haveli',

          'Daman & Diu': 'Daman and Diu',

          'NCT of Delhi':'Delhi'}

df.iloc[:, 1] = df.iloc[:, 1].apply(lambda s: mapper[s] if s in mapper.keys() else s)
# Check to confirm if the names are mapped properly

print(sorted([st['properties']['NAME_1'] for st in india['features']]) == sorted(list(df.iloc[2:, 1].unique())))



data_df = pd.DataFrame(df.iloc[2:, 1].unique(), columns=['State Names in DATA'])

geo_df.merge(data_df, how='outer',

             left_on='State Names in GeoJSON',

             right_on='State Names in DATA')
df_grouped = df.groupby(by='Year')
df_curr = df_grouped.get_group(2001).reset_index(drop=True)



trace = go.Choroplethmapbox(geojson=india,

                            featureidkey='properties.NAME_1',

                            locations=df_curr.loc[1:, 'Name'],

                            z=df_curr.loc[1:, 'LRate_Total'], 

                            zmin=40,

                            zmax=100,

                            colorscale='Viridis',

                            colorbar=dict(title='Percent of State Population',

                                          ticksuffix=' %',

                                          len=0.8,

                                          lenmode='fraction'))



lyt = dict(title='Total Literacy Rate in 2001',

           height=700,

           mapbox_style='white-bg',

           mapbox_zoom=3.4,

           mapbox_center={'lat': 20.5937, 'lon': 78.9629})



fig = go.FigureWidget(data=[trace], layout=lyt)
# Add dropdowns

## 'Total/ Urban/ Rural' dropdown

cat_options = ['Total', 'Rural', 'Urban']

category = widgets.Dropdown(options=cat_options,

                            value='Total',

                            description='Category')



## 'Year' dropdown

year_options = [2001, 2011]

year = widgets.Dropdown(options=year_options,

                        value=2001,

                        description='Year')



# Add Submit button

submit = widgets.Button(description='Submit',

                        disabled=False,

                        button_style='info',

                        icon='check')
def submit_event_handler(args):

    if category.value in ['Total', 'Rural', 'Urban'] and year.value in [2001, 2011]:

        df_curr = df_grouped.get_group(year.value).reset_index(drop=True)

        new_data = df_curr.loc[1:, 'LRate_' + str(category.value)]

        with fig.batch_update():

            fig.data[0].z = new_data

            fig.layout.title = ' '.join([str(category.value), 'Literacy Rate in', str(year.value)])





submit.on_click(submit_event_handler)
container = widgets.HBox([category, year, submit])

widgets.VBox([container, fig])
## Group and Aggregate

df_mean = df.groupby(by=['Year', 'Category']).mean()

df_mean
trace = [go.Bar(name='2001',

                x=['Union Territories', 'States'],

                y=[df_mean.loc[(2001, 'Union Territory'), 'LRate_Total'], df_mean.loc[(2001, 'State'), 'LRate_Total']]),

         go.Bar(name='2011',

                x=['Union Territories', 'States'],

                y=[df_mean.loc[(2011, 'Union Territory'), 'LRate_Total'], df_mean.loc[(2011, 'State'), 'LRate_Total']])]



lyt = dict(barmode='group',

           title='Average Total Literacy Rate (in % Total Population in State/ UT)',

           title_x=0.5, width=600)



fig2 = go.Figure(data=trace, layout=lyt)

fig2.show()
labels = ['Rural', 'Urban']



# Create subplots: use 'domain' type for Pie subplot

fig3 = make_subplots(rows=2, cols=2, specs=[[{'type':'domain'},] * 2, [{'type':'domain'},] * 2])

fig3.add_trace(go.Pie(labels=labels,

                      values=df_mean.loc[(2001, 'Union Territory'), ['Rural_pop', 'Urban_pop']],

                      name="UT"), 1, 1)

fig3.add_trace(go.Pie(labels=labels,

                      values=df_mean.loc[(2001, 'State'), ['Rural_pop', 'Urban_pop']],

                      name="State"), 2, 1)

fig3.add_trace(go.Pie(labels=labels,

                      values=df_mean.loc[(2011, 'Union Territory'), ['Rural_pop', 'Urban_pop']],

                      name="UT"), 1, 2)

fig3.add_trace(go.Pie(labels=labels,

                      values=df_mean.loc[(2011, 'State'), ['Rural_pop', 'Urban_pop']],

                      name="State"), 2, 2)



# Use `hole` to create a donut-like pie chart

fig3.update_traces(hole=.4, hoverinfo="label+percent+name")



fig3.update_layout(

    title_text="Average population in Rural and Urban areas in UTs and States",

    title_x=0.5,

    # Add annotations in the center of the donut pies.

    annotations=[dict(text='2001', x=0.2, y=0.5, font_size=20, showarrow=False),

                 dict(text='2011', x=0.82, y=0.5, font_size=20, showarrow=False)])

fig3.show()