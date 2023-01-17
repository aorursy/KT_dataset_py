import pandas as pd
import numpy as np
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
import plotly.tools as tools
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns 
import calendar
import datetime
init_notebook_mode(connected=True)

path_classified = '../input/olist_classified_public_dataset.csv';
df_classified = pd.read_csv(path_classified)
df_classified.head()
df_classified.describe()
trace1 = go.Histogram(
    x = df_classified['order_products_value'],
    name = 'Order Products Value',
     xbins=dict(
        start=1.0,
        end=50,
        size=0.5
    ),
    marker=dict(
        color='#3CB371'
    ),
    opacity=0.75
)

data = [trace1]

layout = go.Layout(
    title = 'Orders',
    xaxis = dict (
        title='Value'
    ),
    yaxis = dict (
        title='Count'
    ),
    bargap = 0.2,
    bargroupgap = 0.1
)
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig, filename='histogram')
trace1 = go.Box(
    name = 'Order Sellers Qty',
    y = df_classified['order_sellers_qty']
)

trace2 = go.Box(
    name = 'Order Freight Value',
    y = df_classified['order_freight_value']
)

trace3 = go.Box(
    name = 'Order Products Value',
    y = df_classified['order_products_value']
)
data = [trace1, trace2, trace3]
py.offline.iplot(data)
order_by_uf_sorted = df_classified.groupby('customer_state').size().reset_index(name='total_orders')
trace = go.Bar(
    x = order_by_uf_sorted['customer_state'],
    y = order_by_uf_sorted['total_orders'],
    name = 'Orders By UF'
)

data = [trace]
layout = go.Layout(
    title='Total Orders by UF',
    xaxis=dict(
        title='UF',
        tickangle=-45,
        titlefont=dict(
            size=14,
            color='rgb(107,107,107)'
        )
    ),
    yaxis=dict(
        title='Total Orders',
        titlefont=dict(
            size=14,
            color='rgb(107,107,107)'
        )
    )
)

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig, filename='order-by-uf-bar')
df_problems = df_classified[df_classified.most_voted_class == 'problemas_de_entrega'].groupby(['customer_state']).size().reset_index(name='total')

trace = go.Bar(
    x = df_problems['customer_state'],
    y = df_problems['total']
)

data = [trace]
layout = go.Layout(
    title='Delivery Problems By UF',
    xaxis=dict(
        title='UF',
        tickangle=-45,
        titlefont=dict(
            size=14,
            color='rgb(107,107,107)'
        )
    ),
    yaxis=dict(
        title='Total Problems',
        titlefont=dict(
            size=14,
            color='rgb(107,107,107)'
        )
    )
)

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig, filename='order-by-uf-bar')
df_happy_clients = df_classified[(df_classified.most_voted_subclass == 'satisfeito') | (df_classified.most_voted_class == 'satisfeito_com_pedido')].groupby(['customer_state']).size().reset_index(name='total')
trace = go.Bar(
    x = df_happy_clients['customer_state'],
    y = df_happy_clients['total']
)

data = [trace]
layout = go.Layout(
    title='Happy Clients By UF',
    xaxis=dict(
        title='UF',
        tickangle=-45,
        titlefont=dict(
            size=14,
            color='rgb(107,107,107)'
        )
    ),
    yaxis=dict(
        title='Total',
        titlefont=dict(
            size=14,
            color='rgb(107,107,107)'
        )
    )
)

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig, filename='order-by-uf-bar')
month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',
            6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

path_public_v2 = '../input/olist_public_dataset_v2.csv';
df_public_v2 = pd.read_csv(path_public_v2)
df_public_v2['month_code'] = pd.DatetimeIndex(df_public_v2['order_purchase_timestamp']).month
df_public_v2['month'] = pd.DatetimeIndex(df_public_v2['order_purchase_timestamp']).month
df_public_v2['month'] = df_public_v2['month'].apply(lambda x: month_names[x])
df_public_v2['year'] = pd.DatetimeIndex(df_public_v2['order_purchase_timestamp']).year
df_public_v2.head()
path_product_category = '../input/product_category_name_translation.csv';
df_product_category = pd.read_csv(path_product_category)
df_public_v3 = df_public_v2[(df_public_v2.year == 2018)]
df_orders_categories = pd.merge(df_public_v3, df_product_category, on ='product_category_name')
order_by_category = df_orders_categories.groupby('product_category_name_english').size().reset_index(name='total_orders')
order_by_category_top10 = order_by_category.sort_values(by='total_orders', ascending=False).head(10)
trace = go.Bar(
    x = order_by_category_top10.total_orders,
    y = order_by_category_top10.product_category_name_english,
    orientation = 'h',
     marker = dict(
        color='rgba(50, 171, 96, 0.6)',
        line=dict(
            color='rgba(50, 171, 96, 1.0)',
            width=1),
    )
)

data = [trace]
layout = go.Layout(
    title='Top 10 Categories',
    xaxis=dict(
        title='Total Orders',
        titlefont=dict(
            size=14,
            color='rgb(107,107,107)'
        )
    ),
    yaxis=dict(
        title='Categories',
        titlefont=dict(
            size=14,
            color='rgb(107,107,107)'
        )
    ),
    margin=dict(
        l=173,
        r=80,
        t=100,
        b=80,
        pad=0

    )
)

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig, filename='categories-top10')
total_orders_2017 = df_public_v2[df_public_v2.year == 2017].groupby(['month_code','month']).size().reset_index(name='total_orders')
total_orders_2017.sort_values(by='month_code', ascending=True)

total_orders_2018 = df_public_v2[df_public_v2.year == 2018].groupby(['month_code','month']).size().reset_index(name='total_orders')
total_orders_2018.sort_values(by='month_code', ascending=True)
trace1 = go.Scatter(
    x = total_orders_2017.month,
    y = total_orders_2017.total_orders,
    mode = 'lines+markers',
    name = '2017'
)
trace2 = go.Scatter(
    x = total_orders_2018.month,
    y = total_orders_2018.total_orders,
    mode = 'lines+markers',
    name = '2018'
)

layout = go.Layout(
    title='Total Orders By Year',
    xaxis=dict(
        title='Months',
        titlefont=dict(
            size=14,
            color='rgb(107,107,107)'
        )
    ),
    yaxis=dict(
        title='Total Orders',
        titlefont=dict(
            size=14,
            color='rgb(107,107,107)'
        )
    )
)

data = [trace1, trace2]

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig, filename='total-by-year')
categories = df_orders_categories.product_category_name_english.unique()

plt.subplots(figsize =(10,10))
wordcloud = WordCloud(
                        background_color = 'white',
                        width = 512,
                        height = 384
                        ).generate(" ".join(categories))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
