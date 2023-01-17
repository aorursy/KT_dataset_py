# Import Python Modules

import pandas as pd

import plotly.express as px

# Load Data

big_mac_adjusted_index = pd.read_csv('/kaggle/input/the-economists-big-mac-index/output-data/big-mac-adjusted-index.csv')

big_mac_adjusted_index_filtered = big_mac_adjusted_index[['date','name','adj_price']]

# Plot Data

title = 'The Price of a Big Mac Sandwich'

fig = px.line(big_mac_adjusted_index, x="date", y="adj_price",color='name',title=title)

fig.update(layout=dict(xaxis_title='Date',

                       yaxis_title='Price of a Big Mac',

                       legend_orientation="h",

                       showlegend=True))

fig.show()

# Plot Filtered Data

list_of_countries = ['United States','Japan','Russia','China','India']

big_mac_adjusted_index_filtered = big_mac_adjusted_index_filtered[big_mac_adjusted_index_filtered.name.isin(list_of_countries)]

title = 'The Price of a Big Mac Sandwich'

fig = px.line(big_mac_adjusted_index_filtered, x="date", y="adj_price",color='name',title=title)

fig.update(layout=dict(xaxis_title='Date',

                       yaxis_title='Price of a Big Mac',

                       legend_orientation="h",

                       showlegend=True))

fig.show()