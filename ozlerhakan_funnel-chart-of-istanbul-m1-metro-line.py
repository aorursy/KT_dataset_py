import numpy as np 

import pandas as pd 

from plotly import graph_objects as go
df_raw = pd.read_json('/kaggle/input/datastore_search')
df = pd.DataFrame(df_raw.loc['records','result'])
df.info()
df
df.columns = [c.lower().replace(" ", "_") for c in df.columns]
df.fillna(0, inplace=True)
df = df.astype(int)
df.drop(columns='_id', inplace=True)
df.set_index('yil', inplace=True)
df
fig = go.Figure(go.Funnelarea(

    text=[str(y) for y in df.index.tolist()],

    values=df.index.tolist()))





fig.update_layout(

    width = 1000,

    height = 1100,

    title={'text': "Istanbul M1 Metro Line Total Population"})



fig.show()
fig = go.Figure(go.Funnel(

    x=df['m1_hatti'], 

    y=df.index.tolist(),

    textposition = "inside",

    textinfo = "value"))



fig.update_layout(

    width = 1000,

    height = 1100,

    xaxis = go.layout.XAxis(title_text = "Population"),

    yaxis = go.layout.YAxis(title_text = "Years", ticktext=[str(y) for y in df.index.tolist()]),

    title={'text': "Istanbul M1 Metro Line Total Population"})



fig.show()