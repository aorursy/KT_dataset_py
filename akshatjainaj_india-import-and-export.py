import numpy as np # linear algebra

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



india_import = pd.read_csv('/kaggle/input/india-trade-data/2018-2010_import.csv')
india_import.head()
india_export = pd.read_csv('/kaggle/input/india-trade-data/2018-2010_export.csv')
india_export.head()
india_import.info()
india_export.info()
india_import.describe()
india_export.describe()
duplicate_import_rows = india_import[india_import.duplicated(india_import.columns)]

duplicate_import_rows.count()
duplicate_export_rows = india_export[india_export.duplicated(india_export.columns)]

duplicate_export_rows.count()
india_import.drop_duplicates(keep = 'first',inplace = True)
india_import[india_import.country == 'UNSPECIFIED']
india_import['country'].replace(to_replace = 'UNSPECIFIED', value = np.NaN, inplace = True)
india_export[india_export.country == 'UNSPECIFIED']
india_export['country'].replace(to_replace = 'UNSPECIFIED', value = np.NaN, inplace = True)
india_import.dropna(inplace = True , axis = 0)
india_export.dropna(inplace = True , axis = 0)
india_import =  india_import[india_import['value']!=0]

india_export =  india_export[india_export['value']!=0]
india_import.year = pd.Categorical(india_import.year)

india_export.year = pd.Categorical(india_export.year)

len(india_import['Commodity'].unique())
len(india_export['Commodity'].unique())
commodity = pd.DataFrame(india_import['Commodity'].value_counts())

commodity
import_total = india_import.groupby('year').agg({'value':'sum'})

export_total = india_export.groupby('year').agg({'value':'sum'})

import_total['dif'] = export_total.value - import_total.value
import plotly.graph_objects as go



fig = go.Figure()

fig.add_trace(go.Bar(x=import_total.index,

                y=import_total.value,

                name='Import',

                marker_color='rgb(55, 83, 109)'

                ))

fig.add_trace(go.Bar(x=export_total.index,

                y=export_total.value,

                name='Export',

                marker_color='rgb(26, 118, 255)'

                ))

fig.add_trace(go.Bar(x=import_total.index,

                y=import_total.dif,

                name='Deficit',

                marker_color='rgb(27, 120, 210)'

                ))



fig.update_layout(

    title='India Import and Export Yearwise',

    xaxis_tickfont_size=14,

    xaxis = dict(

        title = 'Year',

        titlefont_size=16,

        tickfont_size=14,

    ),

    yaxis=dict(

        title='Value',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15, # gap between bars of adjacent location coordinates.

    bargroupgap=0.1 # gap between bars of the same location coordinate.

)

fig.show()
country_import_total = india_import.groupby('country').agg({'value':'sum'})

country_export_total = india_export.groupby('country').agg({'value':'sum'})

country_import_total = country_import_total.sort_values('value',ascending = False).head(10)

country_export_total = country_export_total.sort_values('value',ascending = False).head(10)
import plotly.graph_objects as go



fig = go.Figure()

fig.add_trace(go.Bar(x=country_import_total.index,

                y=country_import_total.value,

                name='Import',

                marker_color='rgb(55, 83, 109)'

                ))





fig.update_layout(

    title='India Import Country wise',

    xaxis_tickfont_size=14,

    xaxis = dict(

        title = 'Country',

        titlefont_size=16,

        tickfont_size=14,

    ),

    yaxis=dict(

        title='Value',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    

   

)

fig.show()
import plotly.graph_objects as go



fig = go.Figure()

fig.add_trace(go.Bar(x=country_export_total.index,

                y=country_export_total.value,

                name='export',

                marker_color='green'

                ))





fig.update_layout(

    title='India Export Country wise',

    xaxis_tickfont_size=14,

    xaxis = dict(

        title = 'Country',

        titlefont_size=16,

        tickfont_size=14,

    ),

    yaxis=dict(

        title='Value',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    

   

)

fig.show()
fig = go.Figure()



# Add traces

fig.add_trace(go.Scatter(x=import_total.index, y=import_total.value,

                    mode='lines+markers',

                    name='Import'))

fig.add_trace(go.Scatter(x=export_total.index, y=export_total.value,

                    mode='lines+markers',

                    name='Export'))

fig.update_layout(

    title='India Import Export ',

    xaxis_tickfont_size=14,

    xaxis = dict(

        title = 'Year',

        titlefont_size=16,

        tickfont_size=14,

    ),

    yaxis=dict(

        title='Value',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    

   

)

fig.show()
commodity_total_import = india_import.groupby('Commodity').agg({'value':'sum'})

commodity_total_import = commodity_total_import.sort_values('value',ascending = False).head(10)



commodity_total_export = india_export.groupby('Commodity').agg({'value':'sum'})

commodity_total_export = commodity_total_export.sort_values('value',ascending = False).head(10)
fig = go.Figure(data=[go.Bar(y=commodity_total_import.value, x=commodity_total_import.index

            )])

# Customize aspect

fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5, opacity=0.6)

fig.update_layout(title_text='Commodity Import by India')

fig.update_layout(

    xaxis_tickfont_size=14,

    xaxis = dict(

        title = 'Commodity',

        titlefont_size=16,

        tickfont_size=14,

        automargin = True,

    ),

    yaxis=dict(

        title='Value',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    

   

)

fig.show()
fig = go.Figure(data=[go.Bar(y=commodity_total_export.value, x=commodity_total_export.index

            )])

# Customize aspect

fig.update_traces(marker_color='red', marker_line_color='dark red',

                  marker_line_width=1.5, opacity=0.5)

fig.update_layout(title_text='Commodity Export by India')

#layout

fig.update_layout(

    xaxis_tickfont_size=14,

    xaxis = dict(

        title = 'Commodity',

        titlefont_size=16,

        tickfont_size=14,

        automargin = True,

    ),

    yaxis=dict(

        title='Value',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    

   

)

fig.show()
Import_percent = india_import.groupby(['HSCode']).agg({'value':'sum'})

Import_percent = Import_percent.sort_values('value',ascending = False)

Import_percent.head()


labels = Import_percent.index

values = Import_percent.value



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.show()

Export_percent = india_export.groupby(['HSCode']).agg({'value':'sum'})

Export_percent = Export_percent.sort_values('value',ascending = False)

Export_percent.head()
labels = Export_percent.index

values = Export_percent.value



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.show()