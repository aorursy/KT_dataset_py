!pip install --upgrade plotly

from IPython.display import clear_output

clear_output()



import pandas as pd

import plotly

pd.options.display.max_columns = None

import plotly.graph_objects as go

for p in [pd, plotly]:

    print(p.__name__, ': ', p.__version__, sep='')
all_comm = pd.read_csv('../input/UNdata_Export_20190724_225613447.csv')

print(all_comm.dtypes)



all_comm.head()
top_exporters = (all_comm

                 .groupby('Country or Area', as_index=False)

                 .agg({'Trade (USD)':'sum'})

                 .sort_values('Trade (USD)', ascending=False)

                 .head(30))

top_exporters.head(10)

for i, country in enumerate(top_exporters['Country or Area']):

    fig = go.Figure()

    df = all_comm[all_comm['Country or Area']==country]

    df_flat = df.pivot(index='Year', columns='Flow', values='Trade (USD)')

    fig.add_bar(x=df[df['Flow']=='Export']['Trade (USD)'],

                y=df[df['Flow']=='Export']['Year'], 

                orientation='h', name='Exports US$')

    fig.add_bar(x=df[df['Flow']=='Import']['Trade (USD)'].mul(-1),

                y=df[df['Flow']=='Import']['Year'], 

                orientation='h', name='Imports US$')

    fig.add_scatter(x=df_flat['Export'].sub(df_flat['Import']),

                    y=df_flat.index, mode='lines+markers',

                    marker={'color': 'black'},

                    name='Balance')

    fig.layout.barmode = 'relative'

    fig.layout.yaxis = {'showgrid': True}

    fig.layout.title = str(i+1) + ': ' + country + ' Imports and Exports'

    fig.layout.height = 600

    fig.layout.paper_bgcolor = '#E5ECF6'



    fig.show(config={'displayModeBar': False})


