# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('../input/help-international-data'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import warnings

import warnings

warnings.filterwarnings("ignore")

warnings.filterwarnings("always")
#importing all the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import sklearn

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score



from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree
# Reading the csv file

country=pd.read_csv("../input/help-international-data/Country-data.csv")

country.head()
# looking for shape

country.shape
# looking for types

country.info()
# checking the mean

country.describe
#checking columns

country.columns
#checking missing values

country.isnull().sum()
#converting 'exports' in actual values 

country['exports']=round((country['exports']*country['gdpp'])/100,2)
#converting 'health' in actual values

country['health']=round((country['health']*country['gdpp'])/100,2)
#converting 'imports' in actual values

country['imports']=round((country['imports']*country['gdpp'])/100,2)
# checking top 5 values

country.head()
#Perform Analysis for child_mort 

plt.figure(figsize=(15,10))

child_mort=country[['country','child_mort']].sort_values('child_mort',ascending=False).head(10)

ax=sns.barplot(x='country',y='child_mort',data=child_mort)

ax.set(xlabel='',ylabel='child_mortlity Rate')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(15,10))

income=country[['country','income']].sort_values('income',ascending=False).tail(10)

ax=sns.barplot(x='country',y='income',data=income)

ax.set(xlabel='',ylabel='income')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(15,10))

gdpp=country[['country','gdpp']].sort_values('gdpp',ascending=False).tail(10)

ax=sns.barplot(x='country',y='gdpp',data=gdpp)

ax.set(xlabel='',ylabel='GDPP')

plt.xticks(rotation=90)

plt.show()
# performing EDA by making pairplots

sns.set(style="ticks", color_codes=True)

sns.pairplot(country)

plt.show()
# ploting the distplot

plt.figure(figsize = (15,10))

features = country.columns[1:]

for i in enumerate(features):

    plt.subplot(3,3,i[0]+1)

    sns.distplot(country[i[1]])
#plotting boxplots

plt.figure(figsize = (15,10))

features = country.columns[1:]

for i in enumerate(features):

    plt.subplot(3,3,i[0]+1)

    sns.boxplot(country[i[1]])
# capping the lower_end outliers from 'child_mort'

q1=country['child_mort'].quantile(0.01)

country['child_mort'][country['child_mort']<=q1] = q1
#capping upper end outliers from 'exports'

q2=country['exports'].quantile(0.99)

country['exports'][country['exports']>=q2] = q2
#capping upper end outliers from 'health'

q3=country['health'].quantile(0.99)

country['health'][country['health']>=q3] = q3
#capping upper end outliers from 'imports'

q4=country['imports'].quantile(0.99)

country['imports'][country['imports']>=q4] = q4
#capping upper end outliers from 'income'

q5=country['income'].quantile(0.99)

country['income'][country['income']>=q5] = q5
#capping upper end outliers from 'Inflation'

q6=country['inflation'].quantile(0.99)

country['inflation'][country['inflation']>=q6] = q6
#capping upper end outliers from 'life_expec'.

q7=country['life_expec'].quantile(0.99)

country['life_expec'][country['life_expec']>=q7] = q7
#capping upper end outliers from 'gdpp'.-

q9=country['gdpp'].quantile(0.99)

country['gdpp'][country['gdpp']>=q9] = q9
#Calculating the Hopkins statistic

from sklearn.neighbors import NearestNeighbors

from random import sample

from numpy.random import uniform

import numpy as np

from math import isnan

 

def hopkins(X):

    d = X.shape[1]

    #d = len(vars) # columns

    n = len(X) # rows

    m = int(0.1 * n) 

    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)

 

    rand_X = sample(range(0, n, 1), m)

 

    ujd = []

    wjd = []

    for j in range(0, m):

        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)

        ujd.append(u_dist[0][1])

        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)

        wjd.append(w_dist[0][1])

 

    H = sum(ujd) / (sum(ujd) + sum(wjd))

    if isnan(H):

        print(ujd, wjd)

        H = 0

 

    return H
hopkins(country.drop('country',axis = 1))
#dropping 'country' to analyse the data.

country_new=country.drop('country',axis=1)
#scaling

scaler=StandardScaler()

country_scaled=scaler.fit_transform(country_new)

country_scaled.shape
#converting to Dataframe.

country_scaled=pd.DataFrame(country_scaled)

country_scaled.columns=country_new.columns

country_scaled.head()
# calculting the silouette score

ssd=[]

for k in range(2,11):

    kmeans=KMeans(n_clusters=k)  

    kmeans.fit(country_scaled)      #fit the scaled data

    ssd.append([k,silhouette_score(country_scaled,kmeans.labels_)])  #kmeans.labels_

plt.plot(pd.DataFrame(ssd)[0],pd.DataFrame(ssd)[1]) #plotting the curve
# plotting Elbow curve

ssd = []

for k in range(2, 11):

    kmeans = KMeans(n_clusters = k)

    kmeans.fit(country_scaled)   

    ssd.append([k, kmeans.inertia_]) # kmeans.inertia_

    

plt.plot(pd.DataFrame(ssd)[0], pd.DataFrame(ssd)[1])#plotting the curve
# with k=3 making the the model

kmeans=KMeans(n_clusters=3,random_state=100)

kmeans.fit(country_scaled)
#cluster labels assigned

kmeans.labels_
# adding the cluster_labels to the main dataset

country['cluster_label']=kmeans.labels_

country.head()
#counting the cluster_labels in the main dataset

country.cluster_label.value_counts()
# analysing the clusters formed using scatterplot

sns.scatterplot(x = 'child_mort', y = 'gdpp', hue = 'cluster_label', data = country, palette = 'Set1')
# analysing the clusters formed using scatterplot

sns.scatterplot(x = 'income', y = 'gdpp', hue = 'cluster_label', data = country, palette = 'Set1')
# analysing the clusters formed using scatterplot

sns.scatterplot(x = 'child_mort', y = 'income', hue = 'cluster_label', data = country, palette = 'Set1')
# groupping the clusters so formed and finding the mean

country.drop('country',axis = 1).groupby('cluster_label').mean()
#getting country columns

country.columns
#again group by cluster labels and analysing only child_mort,income,gdpp

country.drop(['country', 'exports', 'health', 'imports',

       'inflation', 'life_expec', 'total_fer'],axis=1).groupby('cluster_label').mean().plot(kind='bar')
#finding the countries in cluster=0

country[country['cluster_label']==0]['country']
# cluster profiling , we need to fing the bottom most countries which are in need of aid.

#giving priority to child_mort'over 'income','gdpp' while sorting

country[country['cluster_label']==0].sort_values(by=['child_mort','income','gdpp'],ascending=[False,True,True]).head()
# cluster profiling , we need to fing the bottom most countries which are in need of aid.

#giving priority to 'gdpp' over 'income',child_mort' while sorting

country[country['cluster_label']==0].sort_values(by=['gdpp','income','child_mort'],ascending=[True,True,False]).head()
#Thees are the counytries which are really good and top countries in our data set with least child_mortality and very good gdpp and income

country[country['cluster_label']==1].sort_values(by=['child_mort','income','gdpp'],ascending=[True,False,False]).head()
country_scaled.head()
#single linkage

plt.figure(figsize=(18,15))

country_mergings=linkage(country_scaled,method="single",metric="euclidean")

dendrogram(country_mergings)

plt.show()
#complete linkage

plt.figure(figsize=(18,15))

country_mergings=linkage(country_scaled,method="complete",metric="euclidean")

dendrogram(country_mergings)

plt.show()
#getting number of clusters

cut_tree(country_mergings,n_clusters=3).shape
# adding 'cluster_h_label' which is cluster_id according to hierarchical clustering to the main dataset

cluster_h_label=cut_tree(country_mergings,n_clusters=3).reshape(-1,)

country['cluster_h_label']=cluster_h_label

country.head()
#analysing cluster so formed using scatter plot

sns.scatterplot(x = 'child_mort', y = 'gdpp', hue = 'cluster_h_label', data = country, palette = 'Set1')
#analysing cluster so formed using scatter plot

sns.scatterplot(x = 'income', y = 'gdpp', hue = 'cluster_h_label', data = country, palette = 'Set1')
#analysing cluster so formed using scatter plot

sns.scatterplot(x = 'child_mort', y = 'income', hue = 'cluster_h_label', data = country, palette = 'Set1')
#group the cluster so formed and determining mean

country.drop(['country','cluster_label'],axis = 1).groupby('cluster_h_label').mean()
#cluster profiling where we are getting least five counytries with less gdp,less income, high child_mortality

country[country['cluster_h_label']==0].sort_values(by=['child_mort','income','gdpp'],ascending=[False,True,True]).head()
#cluster profiling where we are getting least five counytries with less gdp,less income, high child_mortality

country[country['cluster_h_label']==0].sort_values(by=['gdpp','income','child_mort'],ascending=[True,True,False]).head()
#cluster profiling where we are getting the countries having good gdp ,least child_mortality

country[country['cluster_h_label']==1].sort_values(by=['child_mort','income','gdpp'],ascending=[True,False,False]).head()
#count the number of elements in each cluster.

country.cluster_h_label.value_counts()