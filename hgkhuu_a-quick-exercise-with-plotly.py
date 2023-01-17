# Import all necessary libraries

import pandas as pd

import numpy as np

import plotly.graph_objects as go

import plotly.express as px



from urllib.request import urlopen

import json



print('Import successful.')
df_solar = pd.read_html('https://github.com/plotly/datasets/blob/master/solar.csv')[0].drop(columns='Unnamed: 0')

df_top3 = df_solar.sort_values(by='Number of Solar Plants', ascending=False).head(3)

df_top3
# Plotting the top 3 states

fig = px.bar(df_top3, x='State', y='Number of Solar Plants',

             title='Top 3 states by number of solar plants',

             color='State',

             hover_name='State',

             hover_data={'State':False}

            )



fig.update_traces(hovertemplate='<b>%{y}</b> solar plants')

fig.update_yaxes(showticklabels=False)

                 



fig.show()
# Plotting all states to compare

fig = px.bar(df_solar.sort_values(by='Number of Solar Plants'), x='State', y='Number of Solar Plants',

             title='Number of Solar plants in each state',

             color='State',

             hover_name='State',

             hover_data={'State':False}

            )



fig.update_traces(hovertemplate='<b>%{y}</b> solar plants')

fig.update_yaxes(showticklabels=False)

                 



fig.show()
# Import data into dataframe

df_tesla = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/tesla-stock-price.csv')

df_tesla.head()
# Transform data before working

df_tesla['date'] = pd.to_datetime(df_tesla['date']) # Convert 'date' column to datetime data type

df_tesla['volume'] = df_tesla['volume'].str.replace(',', '') # Clean 'volume' of any comma components

df_tesla['volume'] = pd.to_numeric(df_tesla['volume']) # Convert 'colume' to numerical data type



df_tesla.head()
# Filter for rows in 2018

year = 2018

df_2018 = df_tesla[df_tesla['date'].dt.year == 2018]

df_2018.shape
# Let's plot with Plotly express

fig = px.line(df_2018, x='date', y='close',

                 title='Closing price of Tesla stock in 2018',

                )



fig.show()
# Let's plot again, but this time with graph_objects. Notice the mechanics are very similar to matplotlib

x = df_2018['date']

y = df_2018['close'].array

closing_mean = [np.average(y) for i in range(len(x))]



fig = go.Figure()



fig.add_trace(go.Scatter(x=x, y=y, mode='markers+lines', name='Closed price',

                         marker=dict(color='blue'), opacity=0.5, x0=min(x)

                        )

             )

fig.add_trace(go.Scatter(x=x, y=closing_mean, mode='lines', name='Average',

                         line=dict(color='black', dash='dash', width=2)

                        )

             )



fig.update_layout(title='Closing price of Tesla stock in 2018 (with graph_objects)',

                  showlegend=False, hovermode="x unified",

                 )

fig.update_yaxes(showticklabels=False)



fig.show()
df_pop = pd.read_html('https://github.com/plotly/datasets/blob/master/minoritymajority.csv', converters={'FIPS':str})[0].drop(columns='Unnamed: 0')

df_pop.head()
df_pop = df_pop.sort_values(by='TOT_POP', ascending=False)

df_pop.groupby('STNAME')['CTYNAME'].aggregate('count').sort_values(ascending=False)
df_top5 = df_pop.head(5)

df_top5
# Let's plot using the top 5 counties



with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:

    counties = json.load(response)

    

fig = px.choropleth(df_top5, geojson=counties, color='STNAME', scope='usa',

                    locations='FIPS', hover_name='CTYNAME', 

                    hover_data={'FIPS':False, 'TOT_POP':':,.0f'},

                   )

fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0}, showlegend=False,

                  title='Top 5 counties with highest population in the US')



fig.show()
