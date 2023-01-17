import numpy as np

import pandas as pd

import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots



from IPython.display import display, HTML

js = "<script>$('.output_scroll').removeClass('output_scroll')</script>"

display(HTML(js))





df = pd.read_csv('../input/covid19-coronavirus/2019_nCoV_data.csv')

# Hide

df.rename(columns={'Date': 'date', 

                     'Id': 'id',

                     'Province/State':'state',

                     'Country':'country',

                     'Lat':'lat',

                     'Long': 'long',

                     'ConfirmedCases': 'confirmed',

                     'Fatalities':'deaths',

                     'ObservationDate': 'obsdate',

                     'Last Update': 'last_upd'

                    }, inplace=True)





# df["obsdate"] = pd.to_datetime(df["obsdate"]).dt.strftime('%m-%d')

# df["last_upd"] = pd.to_datetime(df["last_upd"]).dt.strftime('%m-%d')

# df
# Hide

sea = ['Bangladesh','Cambodia','India','Indonesia','Japan','Malaysia','Philippines','Singapore','Taiwan*','Vietnam']

df.columns

df_sea = df[df.country.isin(sea)]

df_sea.sort_values('date', ascending = False)
# SEA Cases

grouped = df_sea.groupby('date')['date', 'Confirmed', 'Deaths'].sum().reset_index()





fig = px.line(grouped, x="date", y="Confirmed", 

              title="Total Confirmed Cases in Southeast Asia Over Time",

              labels={'obsdate': 'Date', 'Confirmed': 'Confirmed Cases'})



fig.show()



fig = px.line(grouped, x="date", y="Confirmed", 

              title="Total Confirmed Cases in Southease Asia (Logarithmic Scale) Over Time",

              labels={'date': 'Date', 'Confirmed': 'Confirmed Cases (Log)'},

              log_y=True)

fig.show()
fig = px.line(df_sea, x='obsdate', y='Confirmed', color='country',

             title='Confirmed Cases in Southeast Asian Countries',

             labels={'obsdate': 'Date', 'Confirmed': 'Confirmed Cases'})

fig.show()

# Percentage change

import matplotlib.pyplot as plt

import numpy as np



foo = df_sea.sort_values(['country', 'obsdate'], ascending = (True, True))

foo = foo.loc[:,['country','obsdate','Confirmed']]



country = ['Singapore','Indonesia','India','Philippines','Vietnam']

foo = foo.loc[foo['country'].isin(country)]



foo = foo.groupby(['country', 'obsdate']).sum().groupby(level=[0]).cumsum().pct_change().replace(np.inf, 0).cumsum().fillna(0)

foo = foo.reset_index()

foo.sort_values(['obsdate'], ascending=True, inplace=True)



# foo[foo['country'] == 'Philippines']

# foo.to_csv('foo.csv', index=False)



fig = px.line(foo, x='obsdate', y='Confirmed', color='country',

             title='Confirmed Cases in SEA - Cum. Daily Percentage',

             labels={'obsdate': 'Date', 'Confirmed': 'Confirmed Cases'})

fig.show()