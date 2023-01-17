'''

This is my implementation for this tutorial:

https://towardsdatascience.com/data-driven-growth-with-python-part-2-customer-segmentation-5c019d150444

'''







# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install chart_studio
# import libraries

from __future__ import division

from datetime import datetime, timedelta

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns





import chart_studio.plotly as py

import plotly.offline as pyoff

import plotly.graph_objs as go



#inititate Plotly

pyoff.init_notebook_mode()



#load our data from CSV

tx_data=pd.read_csv(r"../input/onlineretail/OnlineRetail.csv", encoding="cp1252")



#convert the string date field to datetime

tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])



#we will be using only UK data

tx_uk = tx_data.query("Country=='United Kingdom'").reset_index(drop=True)
#create a generic user dataframe to keep CustomerID and new segmentation scores

tx_user = pd.DataFrame(tx_data['CustomerID'].unique())

tx_user.columns = ['CustomerID']



#get the max purchase date for each customer and create a dataframe with it

tx_max_purchase = tx_uk.groupby('CustomerID').InvoiceDate.max().reset_index()

tx_max_purchase.columns = ['CustomerID','MaxPurchaseDate']



#we take our observation point as the max invoice date in our dataset

tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days



#merge this dataframe to our new user dataframe

tx_user = pd.merge(tx_user, tx_max_purchase[['CustomerID','Recency']], on='CustomerID')



tx_user.head()



#plot a recency histogram



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

tx_user.head()
tx_user['Recency'].describe()
#Inertia Graph

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
#build 4 clusters for recency and add it to dataframe

kmeans = KMeans(n_clusters=4)

kmeans.fit(tx_user[['Recency']])

tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])



#function for ordering cluster numbers

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
tx_user.describe().transpose()
#get order counts for each user and create a dataframe with it

tx_frequency = tx_uk.groupby('CustomerID').InvoiceDate.count().reset_index()

tx_frequency.columns = ['CustomerID','Frequency']



#add this data to our main dataframe

tx_user = pd.merge(tx_user, tx_frequency, on='CustomerID')



#plot the histogram

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
#k-means

kmeans = KMeans(n_clusters=4)

kmeans.fit(tx_user[['Frequency']])

tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['Frequency']])



#order the frequency cluster

tx_user = order_cluster('FrequencyCluster', 'Frequency',tx_user,True)



#see details of each cluster

tx_user.groupby('FrequencyCluster')['Frequency'].describe()
#calculate revenue for each customer

tx_uk['Revenue'] = tx_uk['UnitPrice'] * tx_uk['Quantity']

tx_revenue = tx_uk.groupby('CustomerID').Revenue.sum().reset_index()



#merge it with our main dataframe

tx_user = pd.merge(tx_user, tx_revenue, on='CustomerID')



#plot the histogram

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
#apply clustering

kmeans = KMeans(n_clusters=4)

kmeans.fit(tx_user[['Revenue']])

tx_user['RevenueCluster'] = kmeans.predict(tx_user[['Revenue']])





#order the cluster numbers

tx_user = order_cluster('RevenueCluster', 'Revenue',tx_user,True)



#show details of the dataframe

tx_user.groupby('RevenueCluster')['Revenue'].describe()
#calculate overall score and use mean() to see details

tx_user['OverallScore'] = tx_user['RecencyCluster'] + tx_user['FrequencyCluster'] + tx_user['RevenueCluster']

tx_user.groupby('OverallScore')['Recency','Frequency','Revenue'].mean()

tx_user.groupby('OverallScore')['Recency' , 'Frequency' ,'Revenue'].mean()
tx_user['Segment'] = 'Low-Value'

tx_user.loc[tx_user['OverallScore']>2,'Segment'] = 'Mid-Value' 

tx_user.loc[tx_user['OverallScore']>4,'Segment'] = 'High-Value' 
#Revenue vs Frequency

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



#Revenue Recency



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



# Revenue vs Frequency

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