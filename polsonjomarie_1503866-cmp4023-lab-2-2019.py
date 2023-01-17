# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

from datetime import datetime, time

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Question 2

df= pd.read_csv("/kaggle/input/Petrol Prices.csv")

df.head(7)
#Question 3

df[-4:]
#Question 4

df['Date']=df.Date.astype('datetime64')

Month = df.Date.dt.month

df['Month']= Month

total = df.groupby('Month').Date.count()

total

#Question 5

total.plot(kind='bar')



#Question 6

pd.to_datetime(df.Date, errors ='coerce')

Month = df.Date.dt.month_name()

Day = df.Date.dt.day

Year = df.Date.dt.year

Timestamp = df.Date.dt.time

#Question 7

#df = pd.DataFrame()

df['Month'] = Month

df['Day'] = Day

df['Year'] = Year

df['Timestamp'] = Timestamp

df
#Question 8

inter_col= df[['Gasolene_87','Gasolene_90','Auto_Diesel','Kerosene','Propane','Butane','HFO','Asphalt','ULSD','Ex_Refinery']]

print(inter_col)
#Question 9

x=inter_col.set_index(df['Date'])

x.plot(kind='line')

#Question 10

Gas=pd.DataFrame(df['Asphalt'])

pctChange = Gas.pct_change(periods=4)

print(pctChange)

pctChange.plot(kind='line')

#Question 11

col= df[['Gasolene_87','Gasolene_90','Month','Day','Year','Timestamp']]

col
#Question 12

t_cluster =  df[['Kerosene','Propane']]

t_cluster.plot(kind='scatter',x='Kerosene',y='Propane')

df_values = t_cluster.iloc[:, :].values

df_values

#Purely integer-location based indexing for selection by position.

#.iloc[] is primarily integer position based (from 0 to length-1 of the axis), but may also be used with a boolean array.
from sklearn.cluster import KMeans

wcss =[]

for i in range (1, 15):

    kmeans= KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300)

    kmeans.fit_predict(df_values)

    wcss.append(kmeans.inertia_)

    plt.plot (wcss, 'ro-', label="WCSS")

    plt.title ("Performing KMeans on DataSet")

    plt.xlabel ("Clusters")

    plt.ylabel ("WCSS")

    plt.show()

    # algoritmn used when the data is unlabel and you try to group them 
#Question 13

kmeans = KMeans(n_clusters=15, init="k-means++", n_init=10, max_iter=300) 

t_cluster["cluster"] = kmeans.fit_predict( df_values )

t_cluster
t_cluster['cluster'].value_counts().plot(kind='bar',title='Sum of cluster')
