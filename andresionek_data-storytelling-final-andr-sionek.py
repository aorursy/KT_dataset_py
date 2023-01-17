import bq_helper

from bq_helper import BigQueryHelper



nyc = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="new_york")



query = """SELECT 

passenger_count,

trip_distance * 1.6 as trip_distance, -- Transformando a distância em metros

pickup_datetime,

dropoff_datetime,

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

and trip_distance * 1.6 <= 5 -- Filtrando somente corridas com menos de 5 km percorridos

and trip_distance * 1.6 > 0 -- Filtrando corridas com distância 0 (possível erro na base)

and tolls_amount = 0 -- Filtrando somente corridas sem pedágio (possivelmente dentro da mesma região)

and rate_code = 1 -- Exluindo corridas para aeroportos e bandeiras especiais

LIMIT 100000

;

"""



data = nyc.query_to_pandas_safe(query, max_gb_scanned=20)
# criando uma coluna de distância em inteiros

data['trip_distance_int'] = data['trip_distance'].round().astype(int)



# Calculando o valor médio por km rodado

valor_por_km = data.groupby(by='trip_distance_int', as_index=False)['total_amount'].mean()

valor_por_km['media_por_km'] = valor_por_km['total_amount'] / valor_por_km['trip_distance_int']

valor_por_km
# Calculando o tempo medio por km rodado

data['duracao'] = (data['dropoff_datetime'] - data['pickup_datetime']).dt.total_seconds() / 3600

duracao_corrida = data.groupby(by='trip_distance_int', as_index=False)['duracao'].mean()

duracao_corrida['velocidade_km_por_h'] = duracao_corrida['trip_distance_int'] / duracao_corrida['duracao']

duracao_corrida
import bq_helper

from bq_helper import BigQueryHelper



nyc = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="new_york")



query = """SELECT 

passenger_count,

trip_distance * 1.6 as trip_distance, -- Transformando a distância em metros

pickup_datetime,

dropoff_datetime,

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

and trip_distance * 1.6 <= 5 -- Filtrando somente corridas com menos de 5 km percorridos

and trip_distance * 1.6 > 0 -- Filtrando corridas com distância 0 (possível erro na base)

and tolls_amount = 0 -- Filtrando somente corridas sem pedágio (possivelmente dentro da mesma região)

and rate_code = 1 -- Exluindo corridas para aeroportos e bandeiras especiais

LIMIT 100000

;

"""



data = nyc.query_to_pandas_safe(query, max_gb_scanned=20)
data['date'] = data['pickup_datetime'].dt.date

data['corridas'] = 1

corridas_dia = data.groupby(by='date', as_index= False)['corridas'].sum()

corridas_dia.head()
corridas_dia['corridas'].mean()
data['dow'] = data['pickup_datetime'].dt.dayofweek

data['hour'] = data['pickup_datetime'].dt.hour



corridas_dia_hora = data.groupby(by=['dow', 'hour'], as_index=False)['corridas'].sum() 

corridas_dia_hora['corridas'] = corridas_dia_hora['corridas'] / data['corridas'].sum() * 100 # Transformando em percentual (pode ser melhor que valores absolutos)0

corridas_dia_hora.pivot('dow', 'hour', 'corridas')
import bq_helper

from bq_helper import BigQueryHelper



nyc = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="new_york")



query = """SELECT 

passenger_count,

trip_distance * 1.6 as trip_distance, -- Transformando a distância em metros

pickup_datetime,

dropoff_datetime,

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

and trip_distance * 1.6 <= 5 -- Filtrando somente corridas com menos de 5 km percorridos

and trip_distance * 1.6 > 0 -- Filtrando corridas com distância 0 (possível erro na base)

and tolls_amount = 0 -- Filtrando somente corridas sem pedágio (possivelmente dentro da mesma região)

and rate_code = 1 -- Exluindo corridas para aeroportos e bandeiras especiais

LIMIT 100000

;

"""



data = nyc.query_to_pandas_safe(query, max_gb_scanned=20)
import pandas as pd

from datashader.utils import lnglat_to_meters as webm

# convertendo as coordenadas pra webmercator

x, y = webm(data.pickup_longitude, data.pickup_latitude)

data['x'] = pd.Series(x)

data['y'] = pd.Series(y)



# filtrando somente manhattan

x_range = (-8242000,-8210000)

y_range = (4965000, 4990000)

data = data[(data['x'] > x_range[0]) & (data['x'] < x_range[1])]

data = data[(data['y'] > y_range[0]) & (data['y'] < y_range[1])]
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

W = 1500 



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
import bq_helper

from bq_helper import BigQueryHelper



nyc = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="new_york")



query = """SELECT 

pickup_datetime,

pickup_longitude,

pickup_latitude,

dropoff_longitude,

dropoff_latitude

FROM

  `bigquery-public-data.new_york.tlc_yellow_trips_2016`

where pickup_longitude is not null

and dropoff_longitude is not null

and trip_distance * 1.6 <= 5 -- Filtrando somente corridas com menos de 5 km percorridos

and trip_distance * 1.6 > 0 -- Filtrando corridas com distância 0 (possível erro na base)

and tolls_amount = 0 -- Filtrando somente corridas sem pedágio (possivelmente dentro da mesma região)

and rate_code = 1 -- Exluindo corridas para aeroportos e bandeiras especiais

LIMIT 10000000

;

"""



data = nyc.query_to_pandas_safe(query, max_gb_scanned=20)
import pandas as pd

from datashader.utils import lnglat_to_meters as webm

# convertendo as coordenadas pra webmercator

x, y = webm(data.pickup_longitude, data.pickup_latitude)

data['x'] = pd.Series(x)

data['y'] = pd.Series(y)



# filtrando somente manhattan

x_range = (-8242000,-8210000)

y_range = (4965000, 4990000)

data = data[(data['x'] > x_range[0]) & (data['x'] < x_range[1])]

data = data[(data['y'] > y_range[0]) & (data['y'] < y_range[1])]
data['dow'] = data['pickup_datetime'].dt.dayofweek

data['hour'] = data['pickup_datetime'].dt.hour



dows = list(range(7)) # dias da semana vao de 0 a 6 

dows
hours = list(range(24)) # dias da semana vao de 0 a 24

hours
# plot wtih datashader - image with black background

import datashader as ds

from datashader import transfer_functions as tf

from functools import partial

from datashader.utils import export_image

from IPython.core.display import HTML, display

from datashader.colors import colormap_select, Greys9

from colorcet import fire, rainbow, bgy, bjy, bkr, kb, kr



background = "black"

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
for dow in dows:

    for hour in hours:

        filtered_data = data[(data['dow'] == dow) & (data['hour'] == hour)]

        create_map(filtered_data, fire, ds.count(),'corridas_dow{}_hour{}'.format(dow, hour))
import bq_helper

from bq_helper import BigQueryHelper



nyc = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="new_york")



query = """SELECT 

passenger_count,

trip_distance * 1.6 as trip_distance, -- Transformando a distância em metros

pickup_datetime,

dropoff_datetime,

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

and trip_distance * 1.6 <= 5 -- Filtrando somente corridas com menos de 5 km percorridos

and trip_distance * 1.6 > 0 -- Filtrando corridas com distância 0 (possível erro na base)

and tolls_amount = 0 -- Filtrando somente corridas sem pedágio (possivelmente dentro da mesma região)

and rate_code = 1 -- Exluindo corridas para aeroportos e bandeiras especiais

LIMIT 100000

;

"""



data = nyc.query_to_pandas_safe(query, max_gb_scanned=20)
data['duracao'] = (data['dropoff_datetime'] - data['pickup_datetime']).dt.total_seconds() / 60

data['custo_patinete'] = data['duracao'].round().astype(int) * 0.5 + 3.5

data.head()
cenario_40pct = data.sample(frac=0.4, random_state=42)

cenario_60pct = data.sample(frac=0.6, random_state=42)

cenario_80pct = data.sample(frac=0.8, random_state=42)
cenario_40pct['custo_patinete'].sum()
cenario_60pct['custo_patinete'].sum()
cenario_80pct['custo_patinete'].sum()