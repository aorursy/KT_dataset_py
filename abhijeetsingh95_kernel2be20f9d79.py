# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

data = pd.read_csv('/kaggle/input/data.csv',encoding='ISO-8859-1')

data.head()
len(data['CustomerID'].unique())
data=data.dropna()
data.drop(['InvoiceNo','StockCode'],axis=1,inplace=True)
import re

def clean(description):

    return re.sub(r'[^\w\s]','',description).lower()

data['Description'] = data['Description'].apply(clean)

data.head()
data['Total'] = data['UnitPrice'] * data['Quantity']
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words='english')
descriptions = vectorizer.fit_transform(data['Description'])

descriptions
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5)

kmeans.fit(descriptions)
data['Product'] = kmeans.labels_

data[data['Product']==3]
customers = pd.DataFrame({'CustomerID':data['CustomerID'].unique()})

customers.head()
from pandasql import sqldf

pysqldf= lambda q:sqldf(q,globals())
q="""select * from 

(

Select CustomerID, 

max(case when Product=0 then tot_spent else 0 end ) over(partition by CustomerID order by null) as Product0,

max(case when Product=1 then tot_spent else 0 end ) over(partition by CustomerID order by null) as Product1,

max(case when Product=2 then tot_spent else 0 end ) over(partition by CustomerID order by null) as Product2,

max(case when Product=3 then tot_spent else 0 end ) over(partition by CustomerID order by null) as Product3,

max(case when Product=4 then tot_spent else 0 end ) over(partition by CustomerID order by null) as Product4

from(

select CustomerID, Product, Sum(Total) as tot_spent from data Group By 1,2) a)b

group by 1,2,3,4,5,6

"""

tst=pysqldf(q)

tst
customers=tst
q="""

select CustomerID, count(Total) as Purchase_count,sum(Total) as Purchase_Total,Min(Total) as Purchase_Min, Max(Total) as Purchase_Max

, avg(Total) as Purchase_Mean

from data Group By 1

"""

tst=pysqldf(q)

tst
customers=pd.merge(customers, tst, how='inner', on='CustomerID')
q="""

select CustomerID, case when Country='United Kingdom' then 0 else 1 end as Foreign_C

from data Group By 1,2

"""

tst=pysqldf(q)

tst
customers=pd.merge(customers, tst, how='inner', on='CustomerID')
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_customers = scaler.fit_transform(customers)
scaled_customers = pd.DataFrame(scaled_customers, columns=customers.columns)
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

cluster_nums = [2,3,4,5,6,7]

scores = []

for cluster_num in cluster_nums:

    kmeans = KMeans(cluster_num)

    kmeans.fit(scaled_customers)

    clusters = kmeans.predict(scaled_customers)

    silhouette = silhouette_score(scaled_customers, clusters)

    scores.append(silhouette)
import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('whitegrid')

plt.ylabel('Silhouette Score')

plt.xlabel('Clusters')

sns.lineplot(x=cluster_nums,y=scores)
kmeans = KMeans(3)

kmeans.fit(scaled_customers)
scaled_customers['Cluster']=kmeans.labels_
plt.figure(figsize=(15,3.5))

sns.heatmap(scaler.inverse_transform(kmeans.cluster_centers_)[:,0:11],annot=True,yticklabels=['Cluster 1','Cluster 2','Cluster 3'],

            xticklabels=scaled_customers.columns.drop('Cluster'))