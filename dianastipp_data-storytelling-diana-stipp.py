print('Hello World!')
'Hello World!'
texto = 'isto é um texto'

texto
inteiro = 12

inteiro
tipofloat = 12.5

tipofloat
tipofloat = inteiro

tipofloat = texto

texto
lista = [1, 'item', [1,2]]

lista

for item in lista:

    print(item)



print('saiu do for')
lista[2]
lista[-2]
idx = 0 



while idx < 10:

    print(idx)

    idx +=1
dicionario = {"id": 252,

              "preço": 35.50,

              "cor": "Azul" 

             }

dicionario
dicionario['cor']
produtos = [{"id": 252,

"preço": 30.00,

"cor": "Azul" 

},

{"id": 253,

"preço": 35.50,

"cor": "Preto" 

},

{"id": 254,

"preço": 99.90,

"cor": "Azul" 

}

]

produtos
for produto in produtos:

    print(produto['cor'])
import matplotlib.pyplot as plt

import seaborn as sns

import folium

import os

import bq_helper

import csv

import requests



from bq_helper import BigQueryHelper



nyc = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="new_york")



query = """SELECT 

pickup_datetime,

dropoff_datetime,

passenger_count,

trip_distance * 1.6 as trip_distance,

pickup_longitude,

pickup_latitude,

dropoff_longitude,

dropoff_latitude,

payment_type,

total_amount

FROM

  `bigquery-public-data.new_york.tlc_yellow_trips_2016`



where pickup_longitude is not null

and dropoff_longitude is not null

and pickup_longitude > - 75

and pickup_longitude < - 72

and pickup_latitude < 42

and pickup_latitude > 40

and trip_distance * 1.6 < 20

and passenger_count = 1

  

LIMIT 100000



;

"""



data = nyc.query_to_pandas_safe(query, max_gb_scanned=20)

data = data[data['pickup_longitude'] < -60]

data.plot(x= 'pickup_longitude', y= 'pickup_latitude', kind='scatter')
import pandas as pd

from datashader.utils import lnglat_to_meters as webm

x, y = webm(data.pickup_longitude, data.pickup_latitude)

data['x'] = pd.Series(x)

data['y'] = pd.Series(y)

data.head()
x_range = (-8242000,-8210000)

y_range = (4965000, 4990000)

data = data[(data['x'] > x_range[0]) & (data['x'] < x_range[1])]

data = data[(data['y'] > y_range[0]) & (data['y'] < y_range[1])]

data.head()
# plot wtih datashader - image with black background

import datashader as ds

from datashader import transfer_functions as tf

from functools import partial

from datashader.utils import export_image

from IPython.core.display import HTML, display

from datashader.colors import colormap_select, Greys9

from colorcet import fire, rainbow, bgy, bjy, bkr, kb, kr



background = "white"

cm = partial(colormap_select, reverse=(background!="black"))

export = partial(export_image, background = background, export_path="export")

display(HTML("<style>.container { width:100% !important; }</style>"))

W = 700 



def create_map(data, cmap, data_agg, export_name='img'):

    pad = (data.x.max() - data.x.min())/50

    x_range, y_range = ((data.x.min() - pad, data.x.max() + pad), 

                             (data.y.min() - pad, data.y.max() + pad))



    ratio = (y_range[1] - y_range[0]) / (x_range[1] - x_range[0])



    plot_width  = int(W)

    plot_height = int(plot_width * ratio)

    if ratio > 1.5:

        plot_height = 550

        plot_width = int(plot_height / ratio)

        

    cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height, x_range=x_range, y_range=y_range)



    agg = cvs.points(data, 'x', 'y', data_agg)

    img = tf.shade(agg, cmap=cmap, how='log')

    return export(img, export_name)

create_map(data, Greys9, ds.count(),'qtd_corridas')
import pandas as pd

data.head(10)
data.describe()
data_filtrado = data[data['pickup_latitude'] > 0 ]

data_filtrado.describe()
data.plot(x='trip_distance', y='total_amount', kind='scatter')
data['passenger_count'].hist(bins=10)
data[(data['trip_distance']<50) & (data['trip_distance']>0)]['trip_distance'].hist()
data['pickup_datetime']
data['pickup_date'] = data['pickup_datetime'].dt.date

data['pickup_hour'] = data['pickup_datetime'].dt.hour

data['pickup_hourminute'] = data['pickup_datetime'].dt.strftime ('%H:%M')
data
corridas_dia = data.groupby(by='pickup_date')['trip_distance'].count()

corridas_hora = data.groupby(by='pickup_hour')['trip_distance'].count()
corridas_dia.plot.line()
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



# Draw a heatmap with the numeric values in each cell

f, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(flights, annot=True, fmt="d", linewidths=.5, ax=ax)
look_up = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',

            6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
data['pickup_month'] = data['pickup_datetime'].dt.month

data['pickup_month_int'] = data['pickup_datetime'].dt.month



data['pickup_month'] = data['pickup_month'].apply(lambda x: look_up[x])

data['pickup_day'] = data['pickup_datetime'].dt.day

data.head()

corridas_dia_mes = data.gropby( by= ['pickup_month', 'pickup_day'])