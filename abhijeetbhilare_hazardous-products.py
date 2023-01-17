import numpy as np

import pandas as pd

import plotly.graph_objs as go

from plotly.offline import iplot

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("/kaggle/input/metal-concentrations/metal-content-of-consumer-products-tested-by-the-nyc-health-department-1.csv")

print(df.shape)

df.head()
df.isnull().sum()
metal_cnt = df.METAL.value_counts()

print(metal_cnt)

data = [go.Bar(

    y = metal_cnt,

    x = metal_cnt.index)]

fig = go.Figure(data=data)

iplot(fig)
country_lst = df.MADE_IN_COUNTRY.unique().tolist()

country_lst
country_lst = country_lst.remove("UNKNOWN OR NOT STATED")
cntry_cnt = df.MADE_IN_COUNTRY.value_counts()

print(metal_cnt)

data = [go.Bar(

    y = cntry_cnt,

    x = cntry_cnt.index)]

fig = go.Figure(data=data)

iplot(fig)
df_worst_products = df.loc[df.CONCENTRATION > 50000, ["METAL", "MADE_IN_COUNTRY", "MANUFACTURER"]]

df_worst_products.shape
metal_cnt = df_worst_products.MADE_IN_COUNTRY.value_counts()

print(metal_cnt)

data = [go.Bar(

    y = metal_cnt,

    x = metal_cnt.index)]

fig = go.Figure(data=data)

iplot(fig)
metal_cnt = df_worst_products.MANUFACTURER.value_counts()

print(metal_cnt)

data = [go.Bar(

    y = metal_cnt,

    x = metal_cnt.index)]

fig = go.Figure(data=data)

iplot(fig)