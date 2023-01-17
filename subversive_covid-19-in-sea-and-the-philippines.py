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



import warnings

warnings.filterwarnings('ignore')



df = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')

df.tail()
# Hide

df.rename(columns={ 

#                      'Id': 'id',

                     'Province/State':'state',

                     'Country/Region':'country',

                     'Lat':'lat',

                     'Long': 'long',

                     'Confirmed': 'confirmed',

                     'Deaths':'deaths',

                     'Recovered': 'recovered',

                     'Date': 'date',

#                      'Last Update': 'last_upd'

                    }, inplace=True)





# df["obsdate"] = pd.to_datetime(df["obsdate"]).dt.strftime('%m-%d')

# df["last_upd"] = pd.to_datetime(df["last_upd"]).dt.strftime('%m-%d')

df.date = pd.to_datetime(df.date)

# df.date = df.date.dt.strftime('%m-%d')

df
# Hide

sea = ['Bangladesh','Cambodia','India','Indonesia','Japan','Malaysia','Philippines','Singapore','Taiwan*','Vietnam']

df.columns

df_sea = df[df.country.isin(sea)]

# df_sea.sort_values('date', ascending = False)

df_sea.groupby('country')['confirmed'].max()
# SEA Cases

grouped = df_sea.groupby('date')['confirmed'].sum().reset_index()





fig = px.line(grouped, x="date", y="confirmed", 

              title="Total Confirmed Cases in Southeast Asia Over Time",

              labels={'date': 'Date', 'confirmed': 'Confirmed Cases'})



fig.show()



fig = px.line(grouped, x="date", y="confirmed", 

              title="Total Confirmed Cases in Southease Asia (Logarithmic Scale) Over Time",

              labels={'date': 'Date', 'confirmed': 'Confirmed Cases (Log)'},

              log_y=True)

fig.show()
fig = px.line(df_sea, x='date', y='confirmed', color='country',

             title='Confirmed Cases in Southeast Asian Countries',

             labels={'date': 'Date', 'confirmed': 'Confirmed Cases'})

fig.show()
# Percentage change

import matplotlib.pyplot as plt

import numpy as np



foo = df_sea.sort_values(['country', 'date'], ascending = (True, True))

foo = foo.loc[:,['country','date','confirmed']]



country = ['Singapore','Indonesia','India','Philippines','Vietnam']

foo = foo.loc[foo['country'].isin(country)]



foo = foo.groupby(['country', 'date']).sum().groupby(level=[0]).cumsum().pct_change().replace(np.inf, 0).cumsum().fillna(0)

foo = foo.reset_index()

foo.sort_values(['date'], ascending=True, inplace=True)



# foo[foo['country'] == 'Philippines']

# foo.to_csv('foo.csv', index=False)



fig = px.line(foo, x='date', y='confirmed', color='country',

             title='Confirmed Cases in SEA - Cum. Daily Percentage',

             labels={'date': 'Date', 'confirmed': 'Confirmed Cases'})

fig.show()
import matplotlib.pyplot as plt

import matplotlib

import squarify



df_sea_grouped = df_sea.groupby('country')['confirmed'].max()

df_sea_grouped = pd.DataFrame(df_sea_grouped)

df_sea_grouped



cmap = matplotlib.cm.jet

mini=min(df_sea_grouped.confirmed)

maxi=max(df_sea_grouped.confirmed)

norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)

colors = [cmap(norm(value)) for value in df_sea_grouped.confirmed]





plt.figure(figsize=(10,10))

squarify.plot(sizes=df_sea_grouped.confirmed, label=df_sea_grouped.index, alpha=0.8, color=colors)

fig.data[0].textinfo = 'label+value'

plt.axis('off')

plt.show()

print(df_sea_grouped.sort_values('confirmed',ascending=False))