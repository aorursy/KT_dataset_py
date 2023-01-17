import pandas as pd

import plotly.express as px
df = pd.read_json("../input/museos-tenerife-json-cabildo-canarias/museos1.json")

df
fig_px = px.scatter_mapbox(df, lat="Latitud", lon="Longitud",

                           hover_name="Nombre",

                           zoom=11, height=300)

fig_px.update_layout(mapbox_style="open-street-map",

                     margin={"r":0,"t":0,"l":0,"b":0})



fig_px.show()
fig_px.update_traces(marker={"size": [10 for x in df]})