# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import geopandas as gpd
import warnings
warnings.simplefilter("ignore")
from scipy.special import boxcox
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import plotly
import plotly.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
from plotly.graph_objs import Scatter, Figure, Layout
from plotly import tools

# Any results you write to the current directory are saved as output.
DIVIDER = '\n------------------------------------------\n'
import os
print(os.listdir("../input"))

for csv in os.listdir("../input"):
    df = pd.read_csv('../input/'+csv)
    print(DIVIDER + csv)
    df.info()
df_payment = pd.read_csv('../input/olist_public_dataset_v2_payments.csv')
print("Total unique ID : {}".format(df_payment.shape[0]-len(df_payment['order_id'].unique())))

print(DIVIDER)
duplicates = df_payment.groupby('order_id').count().sort_values(by='installments',ascending=False)['installments']
print("Top duplicate ID : {}".format(duplicates.head()))

df_payment = df_payment.drop_duplicates('order_id',False)
df_public = pd.read_csv('../input/olist_public_dataset_v2.csv')
df_payment = pd.read_csv('../input/olist_public_dataset_v2_payments.csv')
df_translate = pd.read_csv('../input/product_category_name_translation.csv')

df = pd.merge(df_public,
                 df_payment[['order_id', 'installments', 'sequential', 'payment_type', 'value']],
                 on='order_id')

df = pd.merge(df,
                 df_translate[['product_category_name', 'product_category_name_english']],
                 on='product_category_name')

missing_df = df.isnull().sum()
for i in range(len(missing_df)):
    print("Missing rows in {} : {}".format(missing_df.index[i], missing_df.values[i]))

df.head()
geo = pd.read_csv("../input/geolocation_olist_public_dataset.csv").sample(n=5000)

data = [go.Scattermapbox(
    lon = geo['lng'],
    lat = geo['lat'],
    marker = dict(
        size = 5,
        color = 'green',
    ))]

layout = dict(
        title = 'Brazilian E-Commerce Geolocation',
        mapbox = dict(
            accesstoken = 'pk.eyJ1IjoiaG9vbmtlbmc5MyIsImEiOiJjam43cGhpNng2ZmpxM3JxY3Z4ODl2NWo3In0.SGRvJlToMtgRxw9ZWzPFrA',
            center= dict(lat=-22,lon=-43),
            bearing=10,
            pitch=0,
            zoom=2,
        )
    )
fig = dict( data=data, layout=layout )
iplot( fig, validate=False)
df_geo = pd.read_csv('../input/geolocation_olist_public_dataset.csv')

df['customer_state'] = df['customer_state'].apply(lambda x : x.lower())
df['customer_city'] = df['customer_city'].apply(lambda x : x.lower())
geo_state = df_geo.groupby('state')['lat','lng'].mean().reset_index()
geo_state['state'] = geo_state['state'].apply(lambda x : x.lower())
geo_city = df_geo.groupby('city')['lat','lng'].mean().reset_index()
geo_city['city'] = geo_city['city'].apply(lambda x : x.lower())
geo_city.rename(columns={'lat': 'c_lat','lng':'c_lng'}, inplace=True)

missing_geo = geo_state.isnull().sum()
for i in range(len(missing_geo)):
    print("Missing rows in {} : {}".format(missing_geo.index[i], missing_geo.values[i]))

df = pd.merge(df, geo_state, how='left', left_on='customer_state',right_on='state')
df = pd.merge(df, geo_city,how='left',left_on='customer_city',right_on='city')

df.head()    
data = [go.Scattermapbox(
    lon = geo_state['lng'],
    lat = geo_state['lat'],
    text = geo_state['state'],
    marker = dict(
        size = 20,
        color = 'Tomato',
    ))]

layout = dict(
        title = 'Brazil State Recalculate Coordinate',
        mapbox = dict(
            accesstoken = 'pk.eyJ1IjoiaG9vbmtlbmc5MyIsImEiOiJjam43cGhpNng2ZmpxM3JxY3Z4ODl2NWo3In0.SGRvJlToMtgRxw9ZWzPFrA',
            center= dict(lat=-22,lon=-43),
            bearing=10,
            pitch=0,
            zoom=2,
        )
    )
fig = dict( data=data, layout=layout )
iplot( fig, validate=False)
data = [go.Scattermapbox(
    lon = geo_city['c_lng'],
    lat = geo_city['c_lat'],
    text = geo_city['city'],
    marker = dict(
        size = 2,
        color = 'Green',
    ))]

layout = dict(
        title = 'Brazil Cities Recalculate Coordinate',
        mapbox = dict(
            accesstoken = 'pk.eyJ1IjoiaG9vbmtlbmc5MyIsImEiOiJjam43cGhpNng2ZmpxM3JxY3Z4ODl2NWo3In0.SGRvJlToMtgRxw9ZWzPFrA',
            center= dict(lat=-22,lon=-43),
            bearing=10,
            pitch=0,
            zoom=2,
        )
    )
fig = dict( data=data, layout=layout )
iplot( fig, validate=False)
df['fare']=df['value']-df['order_products_value']
df['extra']=df['fare']-df['order_freight_value']
print("Orders that paying extra : {}".format(df[df['extra']>0.1].count()['order_id']))
print("Orders that paying less : {}".format(df[df['extra']< -0.1].count()['order_id']))

#print(DIVIDER)
overpaid = df.sort_values(by='extra',ascending=False)[['order_id','extra']].head(30)
#print("Top overpaid orders :\n {}".format(overpaid.head()))

#print(DIVIDER)
underpaid = df.sort_values(by='extra',ascending=True)[['order_id','extra']].head(30)
#print("Top under orders :\n {}".format(underpaid.head()))

trace0 = go.Bar(
    y=overpaid['order_id'],
    x=overpaid['extra'],
    name='Overpaid amount',
    marker=dict(color='rgb(49,130,189)'),
    orientation = 'h'
)
trace1 = go.Bar(
    y=underpaid['order_id'],
    x=underpaid['extra'],
    name='Underpaid amount',
    marker=dict(color='rgb(204,204,204)'),
    orientation = 'h'
)
fig = tools.make_subplots(rows=2, cols=1, print_grid=False)
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)

fig['layout'].update(height=800, width=1000, title='Number of installments',
                     legend=dict(orientation="h"))
iplot(fig)
sns.set(style="whitegrid")
fig = plt.figure(figsize=(5,5))
sns.boxplot(y="extra", data=df)
def limit_extra(x):
    if x>1000:
        return 1000
    elif x< -1000:
        return -1000
    else:
        return x
    
df['extra'] = df['extra'].apply(limit_extra)
sns.boxplot(y="extra", data=df)
city_spend = df.groupby(['customer_city','c_lng','c_lat'])['order_products_value'].sum().to_frame().reset_index()
city_freight = df.groupby(['customer_city','c_lng','c_lat'])['order_freight_value'].mean().reset_index()
state_spend = df.groupby(['customer_state','lng','lat'])['order_products_value'].sum().to_frame().reset_index()
state_freight = df.groupby(['customer_state','lng','lat'])['order_freight_value'].mean().reset_index()
state_freight['text'] = 'state :' + state_freight['customer_state'] + ' | Freight: ' + state_freight['order_freight_value'].astype(str)

data = [go.Scattergeo(
    lon = state_spend['lng'],
    lat = state_spend['lat'],
    text = state_freight['text'],
    marker = dict(
        size = state_spend['order_products_value']/3000,
        sizemin = 5,
        color= state_freight['order_freight_value'],
        colorscale= 'Reds',
        cmin = 20,
        cmax = 50,
        line = dict(width=0.1, color='rgb(40,40,40)'),
        sizemode = 'area'
    ),
    name = 'State'),
    go.Scattergeo(
    lon = city_spend['c_lng'],
    lat = city_spend['c_lat'],
    text = city_freight['order_freight_value'],
    marker = dict(
        size = city_spend['order_products_value']/1000,
        sizemin = 2,
        color= city_freight['order_freight_value'],
        colorscale= 'Blues',
        reversescale=True,
        cmin = 0,
        cmax = 80,
        #colorscale = 'RdBu',
        line = dict(width=0.1, color='rgb(40,40,40)'),
        sizemode = 'area'
    ),
    name = 'City')]

layout = dict(
        title = 'Brazilian E-commerce Order and Freight Values (Click legend to toggle traces)',
        showlegend = True,
        autosize=True,
        width = 900,
        height = 600,
        geo = dict(
            scope = "south america",
            projection = dict(type='winkel tripel', scale = 1.6),
            center = dict(lon=-47,lat=-22),
            showland = True,
            showcountries= True,
            showsubunits=True,
            landcolor = 'rgb(155, 155, 155)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        )
    )

fig = dict( data=data, layout=layout )
iplot( fig, validate=False)
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_aproved_at'] = pd.to_datetime(df['order_aproved_at'])
df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
df['delay'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.total_seconds() / (3600 * 24)
df['deliver'] = (df['order_delivered_customer_date'] - df['order_aproved_at']).dt.total_seconds() / (3600 * 24)
df['delay'] = df['delay'].fillna(0)
df['deliver'] = df['deliver'].fillna(0)
print(df['delay'].sort_values(ascending=False).head())
sns.kdeplot(df['delay'])
sns.kdeplot(df['deliver'])
fig = plt.figure(figsize=(20,10))
sns.boxplot(x="customer_state", y="deliver", data=df, palette="Set3")
fig = plt.figure(figsize=(20,10))
sns.boxplot(x="customer_state", y="delay", data=df, palette="Set3")
def lim_deliver(x):
    if x>120:
        return 120
    else:
        return x
def lim_delay(x):
    if x>100:
        return 100
    elif x < -100:
        return -100
    else:
        return x
df['deliver']=df['deliver'].apply(lambda x :lim_deliver(x))
df['delay']=df['delay'].apply(lambda x :lim_delay(x))

fig = plt.figure(figsize=(20,10))
sns.boxplot(x="state", y="delay", data=df, palette="Set3")
sns.boxplot(x="state", y="deliver", data=df, palette="Set3")

city_deliver = df.groupby(['city','c_lng','c_lat'])['deliver'].mean().reset_index()
city_delay = df.groupby(['city','c_lng','c_lat'])['delay'].mean().reset_index()
state_deliver = df.groupby(['state','lng','lat'])['deliver'].mean().reset_index()
state_delay = df.groupby(['state','lng','lat'])['delay'].mean().reset_index()
state_deliver['text'] = 'Deliver duration: ' + state_deliver['deliver'].astype(str) + '| Delay:' + state_delay['delay'].astype(str)
city_deliver['text'] = 'Deliver duration: ' + city_deliver['deliver'].astype(str) + '| Delay:' + city_delay['delay'].astype(str)

data = [go.Scattergeo(
    lon = state_deliver['lng'],
    lat = state_deliver['lat'],
    text = state_deliver['text'],
    marker = dict(
        size = state_deliver['deliver']*20,
        sizemin = 1,
        color= state_delay['delay'],
        colorscale= 'Reds',
        cmin = -30,
        cmax = 0,
        line = dict(width=0.1, color='rgb(40,40,40)'),
        sizemode = 'area'
    ),
    name = 'state'),
    go.Scattergeo(
    lon = city_deliver['c_lng'],
    lat = city_deliver['c_lat'],
    text = city_deliver['text'],
    marker = dict(
        size = (city_deliver['deliver']+3),
        sizemin = 2,
        color= city_delay['delay'],
        colorscale= 'Blues',
        reversescale=True,
        cmin = -50,
        cmax = 50,
        line = dict(width=0.1, color='rgb(40,40,40)'),
        sizemode = 'area'
    ),
    name = 'city')]

layout = dict(
        title = 'Brazilian E-commerce Delivery and Delay (Click legend to toggle traces)',
        showlegend = True,
        autosize=True,
        width = 900,
        height = 600,
        geo = dict(
            scope = "south america",
            projection = dict(type='winkel tripel', scale = 1.6),
            center = dict(lon=-47,lat=-22),
            showland = True,
            showcountries= True,
            showsubunits=True,
            landcolor = 'rgb(155, 155, 155)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        )
    )
fig = dict( data=data, layout=layout )
iplot( fig, validate=False)
city_score = df.groupby(['city','c_lng','c_lat'])['review_score'].mean().reset_index()
state_score = df.groupby(['state','lng','lat'])['review_score'].mean().apply(lambda x: x-3).reset_index()
state_delay = df.groupby(['state','lng','lat'])['delay'].mean().abs().apply(lambda x: x-6).reset_index()
#print(state_score)

fig,ax = plt.subplots(1,2,figsize=(16,9))
sns.barplot(x="state", y="review_score", data=state_score, ax=ax[0])
sns.barplot(x="state", y="delay", data=state_delay, ax=ax[1])
ax[0].set(xlabel='state code', ylabel='(review_score - 3)')
ax[1].set(xlabel='state code', ylabel='(in_advance - 6)')
fig.show()

category_installments = df.groupby('product_category_name_english')['installments'].mean().sort_values(ascending=False).reset_index()
#print(category_installments)
trace1 = go.Bar(
    y=category_installments['product_category_name_english'],
    x=category_installments['installments'],
    name='Underpaid amount',
    marker=dict(color='rgb(204,204,204)'),
    orientation = 'h'
)
fig = dict(data=[trace1])
iplot(fig)