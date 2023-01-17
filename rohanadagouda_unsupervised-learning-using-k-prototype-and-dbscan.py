# Importing required packages

import numpy as np

import pandas as pd 

import random

from kmodes.kprototypes import KPrototypes

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

import os

print(os.listdir("../input"))
#Reading the Data

Data = pd.read_csv('../input/DelayedFlights.csv')

#Dropping the index column

Data= Data.drop(Data.columns[0], axis=1)
#Sanity Check for Negative values in ArrTime, CRSEArrivalTime, DepTime and CRSEDepTime

print(all(i < 0 for i in Data['ArrTime']))

print(all(i < 0 for i in Data['CRSArrTime']))

print(all(i < 0 for i in Data['DepTime']))

print(all(i < 0 for i in Data['CRSDepTime']))
#Converting Categorical Variables to object type

Data[Data.columns[0:4]]=Data[Data.columns[0:4]].astype(object)

Data[Data.columns[8:10]]=Data[Data.columns[8:10]].astype(object)

Data[Data.columns[21:24]]=Data[Data.columns[21:24]].astype(object)
# Renaming the labels within the variables

Data['Cancelled']=Data['Cancelled'].replace([1,0],["Cancelled","Not Cancelled"])

Data['Diverted']=Data['Diverted'].replace([1,0],["Diverted","Not Diverted"])
#Checking for null values

Data.isna().sum()
#Null value Imputation using Interpolation Method

Data['ActualElapsedTime']=Data['ActualElapsedTime'].interpolate(method='linear',limit_direction ='both',axis=0) #Interploation

Data['CRSElapsedTime']=Data['CRSElapsedTime'].interpolate(method='linear',limit_direction ='both',axis=0) #Interploation

Data['AirTime']=Data['AirTime'].interpolate(method='linear',limit_direction ='both',axis=0) #Interploation

Data['ArrTime']=Data['ArrTime'].interpolate(method='linear',limit_direction ='both',axis=0) #Interploation

Data['ArrDelay']=Data['ArrDelay'].interpolate(method='linear',limit_direction ='both',axis=0) #Interploation

Data['CarrierDelay']=Data['CarrierDelay'].interpolate(method='linear',limit_direction ='both',axis=0) #Interploation

Data['WeatherDelay']=Data['WeatherDelay'].interpolate(method='linear',limit_direction ='both',axis=0) #Interploation

Data['NASDelay']=Data['NASDelay'].interpolate(method='linear',limit_direction ='both',axis=0) #Interploation

Data['LateAircraftDelay']=Data['LateAircraftDelay'].interpolate(method='linear',limit_direction ='both',axis=0) #Interploation

Data['SecurityDelay']=Data['SecurityDelay'].interpolate(method='linear',limit_direction ='both',axis=0) #Interploation

Data['TaxiIn']=Data['TaxiIn'].interpolate(method='linear',limit_direction ='both',axis=0) #Interploation

Data['TaxiOut']=Data['TaxiOut'].interpolate(method='linear',limit_direction ='both',axis=0) #Interploation

Data.isna().sum()
#Checking value counts for categorical variables



# Filter categorical variables

num_cols = Data._get_numeric_data().columns

cols = Data.columns

cat_cols = list(set(cols) - set(num_cols))

cat_cols



#Value counts

for col in cat_cols:

    if col in ['DayOfWeek','UniqueCarrier','Month', 'Cancelled','DayofMonth','CancellationCode','Diverted']:

        print(Data[col].value_counts())
#Creating a function to plot Box plot and Histogram

def hist_box_plot(df,feature, fig_num):

    sns.set(color_codes = 'Blue', style="whitegrid")

    sns.set_style("whitegrid", {'axes.grid' : False})

    sns.set_context(rc = {'patch.linewidth': 0.0})

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,3))

    filtered = df.loc[~np.isnan(df[feature]), feature]

    sns.boxplot(filtered, ax = ax1, color = 'steelblue') # boxplot

    sns.distplot(filtered, kde=True, hist=True, kde_kws={'linewidth': 1}, color = 'steelblue', ax = ax2) # histogram

    plt.show()
fig_num = 1        

for col in Data.select_dtypes(include=[np.number]).columns:

    if col in ['DepTime','ActualElapsedTime','CRSElapsedTime','AirTime','ArrDelay','DepDelay','Distance','TaxiIn','TaxiOut']:

        hist_box_plot(Data,col, fig_num)

        fig_num = fig_num + 1
#Creating a function to plot Count plot

def count_plot(df,feature):

    sns.set(color_codes = 'Blue', style="whitegrid")

    sns.set_style("whitegrid", {'axes.grid' : False})

    sns.set_context(rc = {'patch.linewidth': 0.0})

    fig = plt.subplots(figsize=(10,3))

    sns.countplot(x=feature, data=df, color = 'steelblue') # countplot

    plt.show()
# Filter categorical variables

num_cols = Data._get_numeric_data().columns

cols = Data.columns

cat_cols = list(set(cols) - set(num_cols))

cat_cols
for col in cat_cols:

    if col in ['DayOfWeek','UniqueCarrier','Month', 'Cancelled','DayofMonth','CancellationCode','Diverted']:

        count_plot(Data,col)
def biplot(df, x_name, y_name):

    fig, ax = plt.subplots()

    ax.grid(False)

    x = df[x_name]

    y = df[y_name]

    plt.scatter(x,y,c='blue', edgecolors='none',alpha=0.5)

    plt.xlabel(x_name)

    plt.ylabel(y_name)

    plt.title('{x_name} vs. {y_name}'.format(x_name=x_name, y_name=y_name))

    plt.show()
biplot(df=Data,x_name='Month',y_name='ArrDelay')
biplot(df=Data,x_name='DepDelay',y_name='ArrDelay')
biplot(df=Data,x_name='Distance',y_name='ArrDelay')
biplot(df=Data,x_name='Month',y_name='CancellationCode')
# Installation of the Package follows the following steps:

# git clone https://github.com/shakedzy/dython.git

!pip install dython
# Convert all the columns in float to integer for correlation plot as float is not handled

for y in Data.columns:

    if(Data[y].dtype == np.float64):

        Data[y] = Data[y].astype(int)



Data.dtypes



#Ignoring Year, since the data set is for 2008 and UniqueCarrier,FlightNum,and TailNum, since they won't be effective in correlation

Data_Correlation=Data.iloc[:, [1,2,3,4,5,6,7,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]]

Data_Correlation.dtypes
from dython.model_utils import roc_graph

from dython.nominal import associations



def associations_example():

    associations(Data_Correlation,nominal_columns=['Origin','Dest','Cancelled','CancellationCode','Diverted','DayofMonth',

                                           'DayOfWeek','Month'])
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"]=20,20

associations_example()
# Standardizing all the numerical variables

from sklearn import preprocessing

Num_features=Data.select_dtypes(include=[np.number]).columns

Data[Num_features]=preprocessing.MinMaxScaler().fit_transform(Data[Num_features])

Data.head()
#Hopkins Statistic is a way of measuring the cluster tendency of a data set.

from sklearn.neighbors import NearestNeighbors

from random import sample

from numpy.random import uniform

import numpy as np

from math import isnan

 

def hopkins(X):

    d = X.shape[1]

    #d = len(vars) # columns

    n = len(X) # rows

    m = int(0.1 * n) # heuristic from article [1]

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
#Use a random sample of Data for faster computation

Data = Data.sample(20000,random_state=41)

Data.head()

#Resetting the indexs

Data=Data.reset_index(drop=True)

#Rename the levels within in the CancellationCode column

Data['CancellationCode']=Data['CancellationCode'].replace(['N','A','B','C'],[0,1,2,3])

Data['CancellationCode']=Data['CancellationCode'].astype(object)

Data['Cancelled']=Data['Cancelled'].replace(["Cancelled","Not Cancelled"],[1,0])

Data['Cancelled']=Data['Cancelled'].astype(object)

Data.columns
#Checking whether data can be clustered

Num_features =Data.select_dtypes(include=[np.number]).columns

hopkins(Data[Num_features])
#Selection of variables for PCA

Data_pca= Data[['Cancelled','ActualElapsedTime','TaxiOut', 'DepDelay']]

print (Data_pca.dtypes)
#Principal Component

from sklearn.decomposition import PCA

pca = PCA(n_components=3, whiten=True)

Num_features=Data_pca.select_dtypes(include=[np.number]).columns

x=Data_pca[Num_features]

principalComponents = pca.fit_transform(x)



# Cumulative Explained Variance

cum_explained_var = []

for i in range(0, len(pca.explained_variance_ratio_)):

    if i == 0:

        cum_explained_var.append(pca.explained_variance_ratio_[i])

    else:

        cum_explained_var.append(pca.explained_variance_ratio_[i] + 

                                 cum_explained_var[i-1])



print(cum_explained_var)
#Principal Components converted to a Data frame

principalDf  = pd.DataFrame(data = principalComponents

             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])

principalDf.shape
#Concatenating the PCAs with the categorical variable

finalDf_Cat = pd.concat([principalDf, Data_pca['Cancelled']], axis = 1)

finalDf_Cat.head(2)
#Choosing optimal K value

cost = []

X = finalDf_Cat

for num_clusters in list(range(2,7)):

    kproto = KPrototypes(n_clusters=num_clusters, init='Huang', random_state=42,n_jobs=-2,max_iter=15,n_init=50) 

    kproto.fit_predict(X, categorical=[3])

    cost.append(kproto.cost_)



plt.plot(cost)

plt.xlabel('K')

plt.ylabel('cost')

plt.show
# Converting the dataset into matrix

X = finalDf_Cat.as_matrix()
# Running K-Prototype clustering

kproto = KPrototypes(n_clusters=2, init='Huang', verbose=0, random_state=42,max_iter=20, n_init=50,n_jobs=-2,gamma=.25) 

clusters = kproto.fit_predict(X, categorical=[3])
#Visualize K-Prototype clustering on the PCA projected Data

df=pd.DataFrame(finalDf_Cat)

df['Cluster_id']=clusters

print(df['Cluster_id'].value_counts())

sns.pairplot(df,hue='Cluster_id',palette='Dark2',diag_kind='kde')
#Selection of numerical variables for DBSCAN

Data_DBSCAN = Data[['ActualElapsedTime','TaxiOut', 'DepDelay']]
#selection of eps value

from sklearn.neighbors import NearestNeighbors

nbrs=NearestNeighbors().fit(Data_DBSCAN)

distances, indices = nbrs.kneighbors(Data_DBSCAN,20)

kDis = distances[:,10]

kDis.sort()

kDis = kDis[range(len(kDis)-1,0,-1)]

plt.plot(range(0,len(kDis)),kDis)

plt.xlabel('Distance')

plt.ylabel('eps')

plt.show()
#DBSCAN Algorithm

from sklearn.cluster import DBSCAN

dbs_1= DBSCAN(eps=0.035, min_samples=4)

results = dbs_1.fit(Data_DBSCAN).labels_
#Visualize DBSCAN clustering 

df_DBSCAN=Data_DBSCAN

df_DBSCAN['Cluster_id_DBSCAN']=results

print (df_DBSCAN['Cluster_id_DBSCAN'].value_counts())

sns.pairplot(df_DBSCAN,hue='Cluster_id_DBSCAN',palette='Dark2',diag_kind='kde')
#Getting the list of Numerical and Categorical Variables

num_cols = Data._get_numeric_data().columns

print (num_cols)

cols = Data.columns

cat_cols = list(set(cols) - set(num_cols))

cat_cols
#Selection of variables for Kprototype Clustering Algorithm

Data_k= Data[['Cancelled','ActualElapsedTime','TaxiOut', 'DepDelay']]

print (Data_k.dtypes)
#Choosing optimal K value

cost = []

X = Data_k

for num_clusters in list(range(2,7)):

    kproto = KPrototypes(n_clusters=num_clusters, init='Huang', random_state=42,n_jobs=-2,max_iter=15,n_init=50) 

    kproto.fit_predict(X, categorical=[0])

    cost.append(kproto.cost_)



plt.plot(cost)

plt.xlabel('K')

plt.ylabel('cost')

plt.show
# Converting the dataset into matrix

X = Data_k.as_matrix()
# Running K-Prototype clustering

kproto = KPrototypes(n_clusters=2, init='Huang', verbose=0, random_state=42,max_iter=20, n_init=50,n_jobs=-2,gamma=0.15) 

clusters = kproto.fit_predict(X, categorical=[0])
#Visualize K-Prototype clustering 

df_Cancelled=pd.DataFrame(Data_k)

df_Cancelled['Cluster_id_K_Prototype']=clusters

print (df_Cancelled['Cluster_id_K_Prototype'].value_counts())

sns.pairplot(df_Cancelled,hue='Cluster_id_K_Prototype',palette='Dark2',diag_kind='kde')
#Selection of variables for Kprototype Clustering Algorithm

Data_k= Data[['DayofMonth','ActualElapsedTime','TaxiOut', 'DepDelay']]

print (Data_k.dtypes)
# Converting the dataset into matrix

X = Data_k.as_matrix()

# Running K-Prototype clustering

kproto = KPrototypes(n_clusters=2, init='Huang', verbose=0, random_state=42,max_iter=20, n_init=50,n_jobs=-2,gamma=0.15) 

clusters = kproto.fit_predict(X, categorical=[0])

#Visualize K-Prototype clustering

df=pd.DataFrame(Data_k)

df['Cluster_id']=clusters

print(df['Cluster_id'].value_counts())

sns.pairplot(df,hue='Cluster_id',palette='Dark2',diag_kind='kde')
#Selection of variables for Kprototype Clustering Algorithm

Data_k= Data[['DayOfWeek','ActualElapsedTime','TaxiOut', 'DepDelay']]

print(Data_k.dtypes)
# Converting the dataset into matrix

X = Data_k.as_matrix()

# Running K-Prototype clustering

kproto = KPrototypes(n_clusters=2, init='Huang', verbose=0, random_state=42,max_iter=20, n_init=50,n_jobs=-2,gamma=0.15) 

clusters = kproto.fit_predict(X, categorical=[0])

#Visualize K-Prototype clustering 

df=pd.DataFrame(Data_k)

df['Cluster_id']=clusters

print(df['Cluster_id'].value_counts())

sns.pairplot(df,hue='Cluster_id',palette='Dark2',diag_kind='kde')
#Selection of variables for Kprototype Clustering Algorithm

Data_k= Data[['Month','ActualElapsedTime','TaxiOut', 'DepDelay']]

print(Data_k.dtypes)
# Converting the dataset into matrix

X = Data_k.as_matrix()

# Running K-Prototype clustering

kproto = KPrototypes(n_clusters=2, init='Huang', verbose=0, random_state=42,max_iter=20, n_init=50,n_jobs=-2,gamma=0.15) 

clusters = kproto.fit_predict(X, categorical=[0])

#Visualize K-Prototype clustering 

df=pd.DataFrame(Data_k)

df['Cluster_id']=clusters

print(df['Cluster_id'].value_counts())

sns.pairplot(df,hue='Cluster_id',palette='Dark2',diag_kind='kde')
#join DBScan and K-prototype data frames

Clusters=pd.concat([df_Cancelled, df_DBSCAN, Data[['DayOfWeek','DayofMonth','Month']]], axis = 1)

Clusters=Clusters.iloc[:, [0,1,2,3,4,8,9,10,11]]

Clusters.head()
#Verify if randow rows have the same cluster Id between DBScan and K-Prototype

random_rows=Clusters.sample(20,random_state=36)

random_rows.iloc[:, [0,1,2,3,4,5]]
#crosstab table by Day of Week and DBScan Cluster IDs

pd.crosstab(Clusters.DayOfWeek, Clusters.Cluster_id_DBSCAN, margins=True,normalize='columns')
#crosstab table by Day of Week and K-Prototype Cluster IDs

pd.crosstab(Clusters.DayOfWeek, Clusters.Cluster_id_K_Prototype, margins=True,normalize='columns')
#crosstab table by Day of Month and K-Prototype Cluster IDs

pd.crosstab(Clusters.DayofMonth, Clusters.Cluster_id_K_Prototype, margins=True,normalize='columns')
#crosstab table by Day of Month and DBScan Cluster IDs

pd.crosstab(Clusters.DayofMonth, Clusters.Cluster_id_DBSCAN, margins=True,normalize='columns')
#crosstab table by Month and K-Prototype Cluster IDs

pd.crosstab(Clusters.Month, Clusters.Cluster_id_K_Prototype, margins=True,normalize='columns')
#crosstab table by Month and DBScan Cluster IDs

pd.crosstab(Clusters.Month, Clusters.Cluster_id_DBSCAN, margins=True,normalize='columns')