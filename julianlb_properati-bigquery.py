from google.cloud import bigquery

#from google.oauth2 import service_account

import pandas as pd

import numpy as np



import plotly.express as px

from _plotly_future_ import v4_subplots

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

import matplotlib.pyplot as plt

from plotly import tools

from plotly.subplots import make_subplots
client = bigquery.Client()
def populate_table(df):

    

    df['lat'] = df['place'].map(lambda x: x['lat'])

    df['lon'] = df['place'].map(lambda x: x['lon'])

    df['L1'] = df['place'].map(lambda x: x['l1'])

    df['L2'] = df['place'].map(lambda x: x['l2'])

    df['L3'] = df['place'].map(lambda x: x['l3'])

    df['L4'] = df['place'].map(lambda x: x['l4'])

    

    df['operation'] = df['property'].map(lambda x: x['operation'])

    df['type'] = df['property'].map(lambda x: x['type'])

    df['rooms'] = df['property'].map(lambda x: x['rooms'])

    df['bedroms'] = df['property'].map(lambda x: x['bedrooms'])

    df['bathrooms'] = df['property'].map(lambda x: x['bathrooms'])

    df['surface_total'] = df['property'].map(lambda x: x['surface_total'])

    df['surface_covered'] = df['property'].map(lambda x: x['surface_covered'])

    df['price'] = df['property'].map(lambda x: x['price'])

    df['currency'] = df['property'].map(lambda x: x['currency'])

    df['price_period'] = df['property'].map(lambda x: x['price_period'])

    df['title'] = df['property'].map(lambda x: x['title'])

    df['description'] = df['property'].map(lambda x: x['description'])

    

    df = df.drop(['place', 'property'], axis=1)

    

    return df
ventas_capital = client.query("""

            SELECT id, start_date, created_on, place, property

            FROM `properati-dw.public.ads`

            WHERE country = "Argentina"

            AND place.l1 = "Argentina"

            AND place.l2 = 'Capital Federal'

            AND place.l3 != ''

            AND property.operation = 'Venta'

            AND property.type = "Departamento"

            AND property.currency = 'USD'

            AND property.surface_total > 0

            AND property.surface_covered > 0

            AND property.price > 0

            AND property.price < 4000000

            AND TYPE = "Propiedad"

            AND start_date >= "2019-01-01"

            LIMIT 1000

        """

).to_dataframe()





ventas_capital = populate_table(ventas_capital)

ventas_capital['Precio/Metro'] = ventas_capital['price'] / ventas_capital['surface_total']

ventas_capital = ventas_capital.sort_values(by='Precio/Metro', ascending=False)

ventas_capital.head()
fig = px.scatter_mapbox(ventas_capital, lat="lat", lon="lon", hover_data=["L3","L4"],

                        color='Precio/Metro', color_continuous_scale="Magma", zoom=3, height=300)



fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.update_layout( mapbox=dict(center={'lat': -34.6, 'lon': -58.5}, zoom=10))

fig.show()