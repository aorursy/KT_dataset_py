from datetime import datetime, timedelta

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import chart_studio.plotly as py

import plotly.offline as pyoff

import plotly.graph_objs as go
pyoff.init_notebook_mode()
tx_data = pd.read_csv('../input/online-retail-ii-uci/online_retail_II.csv')
tx_data.head(10)
tx_data = tx_data.rename(columns={"Price": "UnitPrice"})

tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])
tx_data['InvoiceDate'].describe()
tx_uk = tx_data.query("Country=='United Kingdom'").reset_index(drop=True)
tx_user = pd.DataFrame(tx_data['Customer ID'].unique())

tx_user.columns = ['Customer ID']
tx_max_purchase = tx_uk.groupby('Customer ID').InvoiceDate.max().reset_index()
tx_max_purchase.columns = ['Customer ID','MaxPurchaseDate']
tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days
tx_user = pd.merge(tx_user, tx_max_purchase[['Customer ID','Recency']], on='Customer ID')
tx_user.head()
tx_user.Recency.describe()
plot_data = [

    go.Histogram(

        x=tx_user['Recency']

    )

]



plot_layout = go.Layout(

        title='Recency'

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
from sklearn.cluster import KMeans





sse={}

tx_recency = tx_user[['Recency']]

for k in range(1, 10):

    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(tx_recency)

    tx_recency["clusters"] = kmeans.labels_

    sse[k] = kmeans.inertia_ 

plt.figure()

plt.plot(list(sse.keys()), list(sse.values()))

plt.xlabel("Number of cluster")

plt.show()
kmeans = KMeans(n_clusters=4)

kmeans.fit(tx_user[['Recency']])

tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])
tx_user.groupby('RecencyCluster')['Recency'].describe()
def order_cluster(cluster_field_name, target_field_name,df,ascending):

    new_cluster_field_name = 'new_' + cluster_field_name

    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()

    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)

    df_new['index'] = df_new.index

    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)

    df_final = df_final.drop([cluster_field_name],axis=1)

    df_final = df_final.rename(columns={"index":cluster_field_name})

    return df_final

tx_user = order_cluster('RecencyCluster', 'Recency',tx_user,False)
tx_frequency = tx_uk.groupby('Customer ID').InvoiceDate.count().reset_index()
tx_frequency.columns = ['Customer ID','Frequency']
tx_frequency.head()
tx_user = pd.merge(tx_user, tx_frequency, on='Customer ID')
tx_user.head()
tx_user.Frequency.describe()
plot_data = [

    go.Histogram(

        x=tx_user.query('Frequency < 1000')['Frequency']

    )

]



plot_layout = go.Layout(

        title='Frequency'

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
sse={}

tx_frequency = tx_user[['Frequency']]

for k in range(1, 10):

    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(tx_frequency)

    tx_frequency["clusters"] = kmeans.labels_

    sse[k] = kmeans.inertia_ 

plt.figure()

plt.plot(list(sse.keys()), list(sse.values()))

plt.xlabel("Number of cluster")

plt.show()
kmeans = KMeans(n_clusters=4)

kmeans.fit(tx_user[['Frequency']])

tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['Frequency']])
tx_user.groupby('FrequencyCluster')['Frequency'].describe()
tx_user = order_cluster('FrequencyCluster', 'Frequency',tx_user,True)
tx_uk['Revenue'] = tx_uk['UnitPrice'] * tx_uk['Quantity']
tx_revenue = tx_uk.groupby('Customer ID').Revenue.sum().reset_index()
tx_revenue.head()
tx_user = pd.merge(tx_user, tx_revenue, on='Customer ID')
tx_user.Revenue.describe()
plot_data = [

    go.Histogram(

        x=tx_user.query('Revenue < 10000')['Revenue']

    )

]



plot_layout = go.Layout(

        title='Monetary Value'

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
import warnings

warnings.filterwarnings("ignore")
sse={}

tx_revenue = tx_user[['Revenue']]

for k in range(1, 10):

    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(tx_revenue)

    tx_revenue["clusters"] = kmeans.labels_

    sse[k] = kmeans.inertia_ 

plt.figure()

plt.plot(list(sse.keys()), list(sse.values()))

plt.xlabel("Number of cluster")

plt.show()
kmeans = KMeans(n_clusters=4)

kmeans.fit(tx_user[['Revenue']])

tx_user['RevenueCluster'] = kmeans.predict(tx_user[['Revenue']])
tx_user = order_cluster('RevenueCluster', 'Revenue',tx_user,True)
tx_user.groupby('RevenueCluster')['Revenue'].describe()
tx_user.head()
tx_user['OverallScore'] = tx_user['RecencyCluster'] + tx_user['FrequencyCluster'] + tx_user['RevenueCluster']
tx_user.groupby('OverallScore')['Recency','Frequency','Revenue'].mean()
tx_user.groupby('OverallScore')['Recency'].count()
tx_user['Segment'] = 'Low-Value'

tx_user.loc[tx_user['OverallScore']>2,'Segment'] = 'Mid-Value' 

tx_user.loc[tx_user['OverallScore']>4,'Segment'] = 'High-Value' 
tx_graph = tx_user.query("Revenue < 50000 and Frequency < 2000")



plot_data = [

    go.Scatter(

        x=tx_graph.query("Segment == 'Low-Value'")['Frequency'],

        y=tx_graph.query("Segment == 'Low-Value'")['Revenue'],

        mode='markers',

        name='Low',

        marker= dict(size= 7,

            line= dict(width=1),

            color= 'blue',

            opacity= 0.8

           )

    ),

        go.Scatter(

        x=tx_graph.query("Segment == 'Mid-Value'")['Frequency'],

        y=tx_graph.query("Segment == 'Mid-Value'")['Revenue'],

        mode='markers',

        name='Mid',

        marker= dict(size= 9,

            line= dict(width=1),

            color= 'green',

            opacity= 0.5

           )

    ),

        go.Scatter(

        x=tx_graph.query("Segment == 'High-Value'")['Frequency'],

        y=tx_graph.query("Segment == 'High-Value'")['Revenue'],

        mode='markers',

        name='High',

        marker= dict(size= 11,

            line= dict(width=1),

            color= 'red',

            opacity= 0.9

           )

    ),

]



plot_layout = go.Layout(

        yaxis= {'title': "Revenue"},

        xaxis= {'title': "Frequency"},

        title='Segments'

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
tx_graph = tx_user.query("Revenue < 50000 and Frequency < 2000")



plot_data = [

    go.Scatter(

        x=tx_graph.query("Segment == 'Low-Value'")['Recency'],

        y=tx_graph.query("Segment == 'Low-Value'")['Revenue'],

        mode='markers',

        name='Low',

        marker= dict(size= 7,

            line= dict(width=1),

            color= 'blue',

            opacity= 0.8

           )

    ),

        go.Scatter(

        x=tx_graph.query("Segment == 'Mid-Value'")['Recency'],

        y=tx_graph.query("Segment == 'Mid-Value'")['Revenue'],

        mode='markers',

        name='Mid',

        marker= dict(size= 9,

            line= dict(width=1),

            color= 'green',

            opacity= 0.5

           )

    ),

        go.Scatter(

        x=tx_graph.query("Segment == 'High-Value'")['Recency'],

        y=tx_graph.query("Segment == 'High-Value'")['Revenue'],

        mode='markers',

        name='High',

        marker= dict(size= 11,

            line= dict(width=1),

            color= 'red',

            opacity= 0.9

           )

    ),

]



plot_layout = go.Layout(

        yaxis= {'title': "Revenue"},

        xaxis= {'title': "Recency"},

        title='Segments'

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
tx_graph = tx_user.query("Revenue < 50000 and Frequency < 2000")



plot_data = [

    go.Scatter(

        x=tx_graph.query("Segment == 'Low-Value'")['Recency'],

        y=tx_graph.query("Segment == 'Low-Value'")['Frequency'],

        mode='markers',

        name='Low',

        marker= dict(size= 7,

            line= dict(width=1),

            color= 'blue',

            opacity= 0.8

           )

    ),

        go.Scatter(

        x=tx_graph.query("Segment == 'Mid-Value'")['Recency'],

        y=tx_graph.query("Segment == 'Mid-Value'")['Frequency'],

        mode='markers',

        name='Mid',

        marker= dict(size= 9,

            line= dict(width=1),

            color= 'green',

            opacity= 0.5

           )

    ),

        go.Scatter(

        x=tx_graph.query("Segment == 'High-Value'")['Recency'],

        y=tx_graph.query("Segment == 'High-Value'")['Frequency'],

        mode='markers',

        name='High',

        marker= dict(size= 11,

            line= dict(width=1),

            color= 'red',

            opacity= 0.9

           )

    ),

]



plot_layout = go.Layout(

        yaxis= {'title': "Frequency"},

        xaxis= {'title': "Recency"},

        title='Segments'

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)