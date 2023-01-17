import pandas as pd

import datetime as dt

import seaborn as sns

import numpy as np

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

%matplotlib inline
df=pd.read_excel("../input/Online Retail.xlsx")
df.columns
df.head()
df.isna().sum()
df.describe()
df=df[ ~df.CustomerID.isna()]
df.isna().sum()
df.describe()
df["total_price"]=df.Quantity*df.UnitPrice
df
df.CustomerID.describe()
df.total_price.describe()
#Top Customers

df.groupby('CustomerID').agg({'total_price':lambda x:sum(x)}).sort_values(ascending=False,by='total_price')
df[df.CustomerID==17448.0]
df.columns
df.Country.unique()
df.Description.unique()
#Unique Customers

df.groupby('CustomerID').agg({'InvoiceNo':lambda x:pd.Series.nunique(x)}).sort_values(ascending=False,by='InvoiceNo')
df[df.CustomerID.isin(df[df.Quantity<0].CustomerID)]
df.InvoiceDate.max()
df.InvoiceDate.min()
(dt.datetime.now()-df.InvoiceDate.min()).days
df.dtypes
current_time=df.InvoiceDate.max()
grp=df.groupby('CustomerID').agg({'InvoiceNo': lambda x:pd.Series.nunique(x),\

                                 'total_price':lambda x: sum(x),

                                 'InvoiceDate':lambda x:(current_time-max(x)).days})
grp=grp.rename(columns={"InvoiceNo":"Frequency",

           "total_price":"Monetry",

           "InvoiceDate":"Recency"})
grp
sns.pairplot(grp)
def score(x,series,recency=False):

    if recency==False:

        if x<=series.quantile(0.25):

            return 1

        elif x<=series.quantile(0.5):

            return 2

        elif x<=series.quantile(0.75):

            return 3

        else:

            return 4

    else:

        if x<=series.quantile(0.25):

            return 4

        elif x<=series.quantile(0.5):

            return 3

        elif x<=series.quantile(0.75):

            return 2

        else:

            return 1

        

grp['R_Score']=grp.Recency.map(lambda x: score(x,grp.Recency,True))

grp['F_Score']=grp.Frequency.map(lambda x: score(x,grp.Frequency))

grp['M_Score']=grp.Monetry.map(lambda x: score(x,grp.Monetry))        
grp
cluster=KMeans(n_clusters=7,random_state=25,max_iter=1000)

cluster.fit(grp[['F_Score',"M_Score","R_Score"]])
labels=cluster.predict(grp[['F_Score', 'R_Score', 'M_Score']])

grp['labels']=labels
grp
score=silhouette_score(grp[['F_Score',"M_Score","R_Score"]],labels)

print('Silhoutte Score:',score)
grp.columns
cluster.cluster_centers_
for k in range(2,20):

    cluster=KMeans(n_clusters=k,random_state=23,max_iter=1000,n_jobs=-1)

    cluster.fit(grp[['F_Score',"M_Score","R_Score"]])

    labels=cluster.predict(grp[['F_Score', 'R_Score', 'M_Score']])

    score=silhouette_score(grp[['F_Score',"M_Score","R_Score"]],labels)

    print('No of Clusters-K:',k,'/','Score:',score)

    
cluster.cluster_centers_
wcss = []

for i in range(1,11):

    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)

    kmeans.fit(grp[['F_Score',"M_Score","R_Score"]])

    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()