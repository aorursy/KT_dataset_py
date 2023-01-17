%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import geopandas as gpd
url_geojson = "/kaggle/input/geojson-departamentos-peru/peru_departamental_simple.geojson"

region_geojson = gpd.read_file(url_geojson)

region_geojson.head()
poblacion =[ #Porblación por depratamento. fuente INEI - CENSO 2017

    157560,

    686728,

    185964,

    1268941,

    358045,

    475068,

    994494,

    731252,

    105862,

    375432,

    786417,

    884928,

    1403555,

    971121,

    9324796,

    606743,

    116743,

    151891,

    160269,

    1471833,

    630648,

    554079,

    296788,

    210592,

    402144,

    ]

region_geojson["HABITANTES_2017"] = poblacion



region_geojson.head()
region_geojson.dtypes
ax = region_geojson.plot(column='HABITANTES_2017',figsize=(15, 15),legend=True,edgecolor=u'gray')
ax = region_geojson.plot(column='HABITANTES_2017',scheme="quantiles",figsize=(15, 15),legend=True, edgecolor=u'gray',cmap='Set2')
ax = region_geojson.plot(column='HABITANTES_2017',scheme="Percentiles",figsize=(15, 15),legend=True, edgecolor=u'gray',cmap='Set2')