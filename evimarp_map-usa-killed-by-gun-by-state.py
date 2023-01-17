import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from plotly.offline import init_notebook_mode, iplot
from plotly import graph_objs as go

init_notebook_mode(connected=True)
gvd = pd.read_csv('../input/gun-violence-data/gun-violence-data_01-2013_03-2018.csv')

gvd.head()
states = pd.read_csv('../input/us-state-county-name-codes/states.csv', index_col=0)

states.head()

gun_killed = (gvd[['state','n_killed']]
              .join(states, on='state')
              .groupby('Abbreviation')
              .sum()['n_killed']
             )
gun_killed.head()

layout = dict(
        title = 'Killed by Gun from 2013-2018 by State',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
data = [go.Choropleth(locationmode='USA-states',
             locations=gun_killed.index.values,
             text=gun_killed.index,
             z=gun_killed.values)]

fig = dict(data=data, layout=layout)

iplot(fig)