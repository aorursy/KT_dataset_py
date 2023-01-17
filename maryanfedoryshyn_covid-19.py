# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
full_table = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

temp = full_table.groupby('ObservationDate')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

temp = temp[temp['ObservationDate']==max(temp['ObservationDate'])].reset_index(drop=True)

temp.style.background_gradient(cmap='Pastel1')
import plotly.express as px

temp = full_table.groupby(['Country/Region', 'ObservationDate', ])['Confirmed', 'Deaths', 'Recovered']

temp = temp.sum().diff().reset_index()



mask = temp['Country/Region'] != temp['Country/Region'].shift(1)



temp.loc[mask, 'Confirmed'] = np.nan

temp.loc[mask, 'Deaths'] = np.nan

temp.loc[mask, 'Recovered'] = np.nan



fig = px.bar(temp.sort_values('ObservationDate'), 

             x="ObservationDate", y="Confirmed", color='Country/Region',title='New cases',)

fig.update_layout(showlegend=False)

fig.show()



###

temp = full_table.groupby(['Country/Region', 'ObservationDate', ])['Confirmed', 'Deaths', 'Recovered']

temp = temp.sum().diff().reset_index()

temp = temp[temp['Country/Region'] != 'US']



mask = temp['Country/Region'] != temp['Country/Region'].shift(1)



temp.loc[mask, 'Confirmed'] = np.nan

temp.loc[mask, 'Deaths'] = np.nan

temp.loc[mask, 'Recovered'] = np.nan



fig = px.bar(temp.sort_values('ObservationDate'), 

             x="ObservationDate", y="Confirmed", color='Country/Region',title='New cases WO USA',)

fig.update_layout(showlegend=False)

fig.show()

###



###

temp = full_table.groupby(['Country/Region', 'ObservationDate', ])['Confirmed', 'Deaths', 'Recovered']

temp = temp.sum().diff().reset_index()

temp = temp[temp['Country/Region'] != 'US']

temp = temp[temp['Country/Region'] != 'France']



mask = temp['Country/Region'] != temp['Country/Region'].shift(1)



temp.loc[mask, 'Confirmed'] = np.nan

temp.loc[mask, 'Deaths'] = np.nan

temp.loc[mask, 'Recovered'] = np.nan



fig = px.bar(temp.sort_values('ObservationDate'), 

             x="ObservationDate", y="Confirmed", color='Country/Region',title='New cases WO USA and France',)

fig.update_layout(showlegend=False)

fig.show()

###



temp = full_table.groupby(['Country/Region', 'ObservationDate', ])['Confirmed', 'Deaths', 'Recovered']

temp = temp.sum().diff().reset_index()

temp = temp[temp['Country/Region'] == 'Ukraine']



mask = temp['Country/Region'] != temp['Country/Region'].shift(1)



temp.loc[mask, 'Confirmed'] = np.nan

temp.loc[mask, 'Deaths'] = np.nan

temp.loc[mask, 'Recovered'] = np.nan



fig = px.bar(temp.sort_values('ObservationDate'), 

             x="ObservationDate", y="Confirmed", color='Country/Region',title='New cases Ukraine',

            )

fig.update_layout(showlegend=False)

fig.show()
temp = full_table.groupby(['ObservationDate', 'Country/Region'])['Confirmed'].sum().reset_index().sort_values('Confirmed', ascending=False)



fig = px.line(temp.sort_values('ObservationDate'), x="ObservationDate", y="Confirmed", color='Country/Region', title='Cases Spread',)

fig.update_layout(showlegend=False)

fig.show()



temp = full_table.groupby(['ObservationDate', 'Country/Region'])['Confirmed'].sum().reset_index().sort_values('Confirmed', ascending=False)

temp = temp[temp['Country/Region'] == 'Ukraine']

fig = px.line(temp.sort_values('ObservationDate'), x="ObservationDate", y="Confirmed", color='Country/Region', title='Cases Spread Ukraine',)

fig.update_layout(showlegend=False)

fig.show()
gdf = full_table.groupby(['ObservationDate', 'Country/Region'])['Confirmed', 'Deaths','Recovered'].max()

gdf = gdf.reset_index()



temp = gdf[gdf['Country/Region']=='Mainland China'].reset_index()

temp = temp.melt(id_vars='ObservationDate', value_vars=['Confirmed', 'Deaths','Recovered'],

                var_name='Case', value_name='Count')

fig = px.bar(temp, x="ObservationDate", y="Count", color='Case', facet_col="Case",

            title='China'#, color_discrete_sequence=[cnf, dth, rec]

            )

fig.show()



temp = gdf[gdf['Country/Region']!='Mainland China'].groupby('ObservationDate').sum().reset_index()

temp = temp.melt(id_vars='ObservationDate', value_vars=['Confirmed', 'Deaths','Recovered'],

                var_name='Case', value_name='Count')

fig = px.bar(temp, x="ObservationDate", y="Count", color='Case', facet_col="Case",

             title='ROW'#, color_discrete_sequence=[cnf, dth, rec]

            )

fig.show()



temp = gdf[gdf['Country/Region']=='Ukraine'].groupby('ObservationDate').sum().reset_index()

temp = temp.melt(id_vars='ObservationDate', value_vars=['Confirmed', 'Deaths','Recovered'],

                var_name='Case', value_name='Count')

fig = px.bar(temp, x="ObservationDate", y="Count", color='Case', facet_col="Case",

             title='Ukraine'#, color_discrete_sequence=[cnf, dth, rec]

            )

fig.show()