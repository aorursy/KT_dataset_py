import warnings

warnings.filterwarnings('ignore')

from datetime import datetime

from dateutil.relativedelta import relativedelta

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns



import sklearn.preprocessing as pp

from sklearn.pipeline import Pipeline

from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, AffinityPropagation, Birch, DBSCAN, OPTICS

from sklearn.mixture import GaussianMixture

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score



%matplotlib inline

pd.set_option('precision', 2)

sns.set_style('whitegrid')
data = pd.read_excel('/kaggle/input/uci-online-retail-ii-data-set/online_retail_II.xlsx')
data.shape
data.head()
data.info()
# Identify the number of NAs in each feature and select only those having NAs

total_NA = data.isnull().sum()[data.isnull().sum() != 0]



# Calculate the percentage of NA in each feature

percent_NA = data.isnull().sum()[data.isnull().sum() != 0]/data.shape[0]



# Summarize our findings in a dataframe

missing = pd.concat([total_NA, percent_NA], axis=1, keys=['Total NAs', 'Percentage']).sort_values('Total NAs', ascending=False)

missing
# Drop transactions with missing Customer ID

data.dropna(axis=0, subset=['Customer ID'], inplace= True)
print('Number of duplicated records: ', data.duplicated(keep='first').sum())
indx = data[data.duplicated(keep='first')].index

data.drop(index = indx, inplace= True)
data[['StockCode']] = data['StockCode'].astype(str)

data[['Customer ID']] = data['Customer ID'].astype(int).astype(str)
data.dtypes.value_counts()
# Summary statistics of categorical variables

data.select_dtypes(include='object').describe().T
# Summary statistics of "InvocieDate" variable

data[['InvoiceDate']].describe().T
# Summary statistics of numeric variables

data.select_dtypes(include= ['int64', 'float64']).describe().transpose()
x = data.Country.apply(lambda x: x if x == 'United Kingdom' else 'Not UK').value_counts().rename('#Customers')

y = (x/data.shape[0]).rename('%Customers')

pd.concat([x, y], axis= 1)
# Drop cancelled transactions

indx = data.Invoice[data.Invoice.str.contains('C') == True].index

data.drop(index= indx, inplace= True)
# Drop transactions with price zero

indx = data.loc[data.Price == 0].index

data.drop(index= indx, inplace= True)
# Amount per transaction which is the product of sale price and quantity

data['Amount'] = data['Price'] * data['Quantity']
# Create new variable for Invoice time in hours

data['Transaction_time'] = data.InvoiceDate.apply(lambda x : x.time().hour)
# Create new variable for Invoice date

data['Transaction_date'] = data.InvoiceDate.apply(lambda x : x.date())

data['Transaction_date'] = data.Transaction_date.apply(lambda x: x.replace(day = 1))
# calculate the no. of months since transaction date .

ref = datetime.strptime('2010-12', '%Y-%m')

data['Mnths_since_purchase'] = data.Transaction_date.apply(lambda x: \

                                        relativedelta(ref,x).years*12 + relativedelta(ref,x).months)

Recency = data.groupby('Customer ID').agg({'Mnths_since_purchase' : 'min'}).copy().rename(columns= {'Mnths_since_purchase':'Recency'})
# Calculate the number of months since the first purchase for each customer

data['First_purchase'] = data['Mnths_since_purchase'].copy()

First_purchase = data.groupby('Customer ID').agg({'First_purchase' : 'max'}).copy().rename(columns= {'Mnths_since_purchase':'First_purchase'})
Frequency = data.groupby(['Customer ID',

                    'Transaction_date']).agg({'Invoice' : 'nunique'}).groupby(['Customer ID']).agg({'Invoice' : 'mean'}).copy().rename(columns= {'Invoice':'Frequency'})
Monetary_value = data.groupby(['Customer ID',

                    'Invoice']).agg({'Amount' : 'sum'}).groupby(['Customer ID']).agg({'Amount' : 'mean'}).copy().rename(columns= {'Invoice':'Frequency',

                                                              'Amount': 'Monetary_value'})
# Calculate Average number of unique items in each transaction for each customer

unique_items = data.groupby(['Customer ID', 'Invoice']).agg({'StockCode': 'nunique'}).groupby(['Customer ID']\

            ).agg({'StockCode':'mean'}).rename(columns={'StockCode': 'Unique_items'})
# Create transformed data for Clustering

data_transformed = pd.concat([Recency, First_purchase, Frequency, Monetary_value,unique_items], axis=1)

data_transformed.describe()
# Plot the distribution of all variables that will be used for model training

fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(9,9))

sns.distplot(data_transformed.Recency, ax= ax[0][0], kde= False)

sns.distplot(data_transformed.First_purchase, ax= ax[0][1], kde= False)

sns.distplot(data_transformed.Frequency, ax= ax[1][0], kde= False)

sns.distplot(data_transformed.Monetary_value, ax= ax[1][1], kde= False)

sns.distplot(data_transformed.Unique_items, ax= ax[2][0], kde= False)
# Define frequency threshold value and drop customers who exceed the threshold

freq_stats = data_transformed['Frequency'].describe()

freq_threshold = freq_stats['mean'] + 3 * freq_stats['std']

indx = data_transformed.loc[data_transformed.Frequency > freq_threshold].index

data_transformed.drop(index = indx, inplace= True)
# Define Monetary value threshold value and drop customers who exceed the threshold

m_stats = data_transformed['Monetary_value'].describe()

m_threshold = m_stats['mean'] + 3 * m_stats['std']

indx = data_transformed.loc[data_transformed.Monetary_value > m_threshold].index

data_transformed.drop(index = indx, inplace= True)
fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(9,9))

sns.distplot(data_transformed.Recency, ax= ax[0][0], kde= False)

sns.distplot(data_transformed.First_purchase, ax= ax[0][1], kde= False)

sns.distplot(data_transformed.Frequency, ax= ax[1][0], kde= False)

sns.distplot(data_transformed.Monetary_value, ax= ax[1][1], kde= False)

sns.distplot(data_transformed.Unique_items, ax= ax[2][0], kde= False)
# Normalize the four variables

scaler = pp.StandardScaler()

data_transformed_scaled = pd.DataFrame(scaler.fit_transform(data_transformed),

                                       columns= data_transformed.columns)
fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(9,9))

sns.distplot(data_transformed_scaled.Recency, ax= ax[0][0], kde= False)

sns.distplot(data_transformed_scaled.First_purchase, ax= ax[0][1], kde= False)

sns.distplot(data_transformed_scaled.Frequency, ax= ax[1][0], kde= False)

sns.distplot(data_transformed_scaled.Monetary_value, ax= ax[1][1], kde= False)

sns.distplot(data_transformed_scaled.Unique_items, ax= ax[2][0], kde= False)
# A function to automate the model fiting and prediction

def model_train(estimator, data, a,b):

    db = []

    ca = []

    sc = []

    bic = []

    aic = []

    n_clusters = {'n_clusters':[]}

    if (estimator == AffinityPropagation)|(estimator == DBSCAN)|(estimator == OPTICS)|(estimator==Birch):

        est = estimator()

        est.fit(data)  

        labels = est.labels_

        if np.unique(est.labels_).shape[0] > 1:

            db.append(davies_bouldin_score(data, labels))

            ca.append(calinski_harabasz_score(data, labels))

            sc.append(silhouette_score(data, labels))

            n_clusters['n_clusters'].append('N/A')

        else:

            n_clusters['n_clusters'].append(np.unique(est.labels_).shape[0])

    

    else:

        for k in range(a, b):

            if estimator == GaussianMixture:

                est = estimator(n_components= k)

                labels = est.fit_predict(data)

            else:

                est = estimator(n_clusters= k)

                est.fit(data)

                labels = est.labels_



            db.append(davies_bouldin_score(data, labels))

            ca.append(calinski_harabasz_score(data, labels))

            sc.append(silhouette_score(data, labels))



        n_clusters['n_clusters'].append(np.argmin(db) + a)

        n_clusters['n_clusters'].append(np.argmax(ca) + a)

        n_clusters['n_clusters'].append(np.argmax(sc) + a)

    return db, ca, sc, labels, n_clusters['n_clusters']
#Plot different measures against No. of clusters for algorithms requiring no. of clusters a priori.

def plot_scores(a,b, db, ca, sc):

    fig, ax = plt.subplots(nrows= 1, ncols=3, figsize=(15,4))

    ax[0].plot(range(a, b), db, "bo-", label= 'Davies_Bouldin_Score')

    ax[1].plot(range(a, b), ca, "rx-", label = 'Calinski_Harabasz_Score')

    ax[2].plot(range(a, b), sc, "g.-", label = 'Silhouette_Score')

    ax[0].set_xlabel("$k$", fontsize=14)

    ax[1].set_xlabel("$k$", fontsize=14)

    ax[2].set_xlabel("$k$", fontsize=14)

    ax[0].set_ylabel('Davies Bouldin Score', fontsize=14)

    ax[1].set_ylabel('Calinski Harabasz Score', fontsize=14)

    ax[2].set_ylabel('Silhouette Score', fontsize=14)

#     plt.legend(loc=(1,0),fontsize=14)

    plt.show()
clusterers = [KMeans, AffinityPropagation, AgglomerativeClustering, Birch,

             DBSCAN, GaussianMixture, OPTICS, SpectralClustering]



Scores ={'Davies_Bouldin_Score': [], 

         'Calinski_Harabasz_Score': [],

         'Silhouette_Score': [],

        'n_clusters': []}



clusterer_names = ['KMeans', 'Affinity Propagation', 'Agglomerative Clustering', 'Birch',

             'DBSCAN', 'Gaussian Mixture Model', 'OPTICS', 'Spectral Clustering']



for i in clusterers:

    db, ca, sc, labels, n_clusters= model_train(i, data_transformed_scaled, 3, 8)



    Scores['Davies_Bouldin_Score'].append(np.min(db))

    Scores['Calinski_Harabasz_Score'].append(np.max(ca))

    Scores['Silhouette_Score'].append(np.max(sc))

    Scores['n_clusters'].append(n_clusters)
models_scores = pd.DataFrame(Scores, index= clusterer_names)

models_scores
models_scores.loc[models_scores.Davies_Bouldin_Score == models_scores.Davies_Bouldin_Score.min()]
models_scores.loc[models_scores.Calinski_Harabasz_Score == models_scores.Calinski_Harabasz_Score.max()]
models_scores.loc[models_scores.Silhouette_Score == models_scores.Silhouette_Score.max()]
def cluster_stats(model, data, data_transformed):

    df = data_transformed.copy()

    df['Cluster'] = pd.Series(model.labels_, name= 'Cluster', index= data_transformed.index)

    df['No._Purchases'] = data.groupby('Customer ID')['Invoice'].count()[df.index]

    df['Total_Amount'] = data.groupby('Customer ID')['Amount'].sum()[df.index]

    cluster_stats = df.groupby('Cluster').agg({'Recency': ['min', 'mean','max'],

                                       'Frequency': ['min', 'mean','max'],

                                       'Monetary_value': ['min', 'mean','max'],

                                       'First_purchase': ['min', 'mean','max'],

                                       'Unique_items': ['min', 'mean','max']}).copy().round(1)

    return cluster_stats, df
def clusters_summary(df, data):

    columns = {'#Customers':[], '#Purchases':[], 'Total_Amount':[]}

    indx =[]

    for i in np.sort(df.Cluster.unique()):

        columns['#Customers'].append(data.iloc[df.loc[df.Cluster == i].index].shape[0])

        columns['#Purchases'].append(df['No._Purchases'].loc[df.Cluster == i].sum())

        columns['Total_Amount'].append(df['Total_Amount'].loc[df.Cluster == i].sum())

        indx.append('Cluster{}'.format(i))

    

    # Synthesis a data frame for cluster summanry

    clusters_summary = pd.DataFrame(data= columns, index = indx)



    clusters_summary['%customers'] = (clusters_summary['#Customers']/df.shape[0])*100

    clusters_summary['%transactions'] = (clusters_summary['#Purchases']/df['No._Purchases'].sum())*100

    clusters_summary['%sales_amount'] = (clusters_summary['Total_Amount']/df['Total_Amount'].sum())*100

    columnsOrder = ['#Customers', '%customers', '#Purchases', '%transactions', 'Total_Amount', '%sales_amount']

    return clusters_summary.reindex(columns=columnsOrder)    
def plot_3d(cluster_stat):

    fig = plt.figure(figsize=(9,7))

    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(cluster_stat['Recency'], cluster_stat['Frequency'], cluster_stat['Monetary_value'],

                     c=cluster_stat['Cluster'], s=60, marker=".",

                         cmap= 'prism', edgecolor= 'k', linewidths= 0.6)

    # produce a legend with the unique colors from the scatter

#     legend1 = ax.legend(*scatter.legend_elements(),

#                         loc="center right", title="Customer Segments",

#               bbox_to_anchor=(0.75, 0, 0.5, 1), fontsize= 12)

#     ax.add_artist(legend1)

    ax.set_xlabel('Recency', fontsize= 12)

    ax.set_ylabel('Frequency', fontsize= 12)

    ax.set_zlabel('Monetary_value', fontsize= 12)
def plot_dist(df, col):

    n= df['Cluster'].nunique()

    mpl.rcParams['figure.figsize'] = (12,12)

    fig, ax = plt.subplots(ncols=2 , nrows= (n//2))

    k = 0

    h=0

    for j in col:

        for i in range(n):

            sns.distplot(df[j][df.Cluster ==i], hist= False, label= 'Cluster{}'.format(i),

                         ax= ax[k][h], kde= True)

        ax[k][h].set_xlabel('{}'.format(j), fontsize= 14)

        h+=1

        if h%2==0:

            h=0

            k +=1
specc_3 = SpectralClustering(n_clusters= 3).fit(data_transformed_scaled)
cluster_stats(specc_3, data, data_transformed)[0]
clusters_summary(cluster_stats(specc_3, data, data_transformed)[1], data)
plot_3d(cluster_stats(specc_3, data, data_transformed)[1])
specc_4 = SpectralClustering(n_clusters= 4).fit(data_transformed_scaled)
cluster_stats(specc_4, data, data_transformed)[0]
clusters_summary(cluster_stats(specc_4, data, data_transformed)[1], data)
plot_3d(cluster_stats(specc_4, data, data_transformed)[1])
kmeans = KMeans(n_clusters= 5, max_iter= 1000, random_state= 42).fit(data_transformed_scaled)
cluster_stats(kmeans, data, data_transformed)[0]
kmeans_cs = clusters_summary(cluster_stats(kmeans, data, data_transformed)[1], data)

kmeans_cs
plot_3d(cluster_stats(kmeans, data, data_transformed)[1])
Scores ={'Davies_Bouldin_Score': [], 

         'Calinski_Harabasz_Score': [],

         'Silhouette_Score': []}



for k in range(4,9):

    kmeans = KMeans(n_clusters= k).fit(data_transformed_scaled)

    Scores['Davies_Bouldin_Score'].append(davies_bouldin_score(data_transformed_scaled, 

                                                               kmeans.labels_))

    Scores['Calinski_Harabasz_Score'].append(calinski_harabasz_score(data_transformed_scaled, 

                                                                     kmeans.labels_))

    Scores['Silhouette_Score'].append(silhouette_score(data_transformed_scaled, 

                                                       kmeans.labels_))
plot_scores(4,9,Scores['Davies_Bouldin_Score'], Scores['Calinski_Harabasz_Score'],

            Scores['Silhouette_Score'])
kmeans = KMeans(n_clusters= 6, max_iter= 1000, random_state= 42).fit(data_transformed_scaled)
kmeans_results = cluster_stats(kmeans, data, data_transformed)[1]

cluster_stats(kmeans, data, data_transformed)[0]
kmeans_cs = clusters_summary(cluster_stats(kmeans, data, data_transformed)[1], data)

kmeans_cs
plot_3d(cluster_stats(kmeans, data, data_transformed)[1])
# Two Pie charts to compare clusters in terms of represented population proportion and total sales

# amount proportion

def func(pct):

    return "{:.1f}%".format(pct)



fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(15, 6), subplot_kw=dict(aspect="equal"))



wedges, text1, autotexts = ax[1].pie(kmeans_cs['%sales_amount'].values,

                                  autopct=lambda pct: func(pct),

                                  textprops=dict(color="w", fontsize= 12))

wedges, text2, autotexts = ax[0].pie(kmeans_cs['%customers'].values,

                                  autopct=lambda pct: func(pct),

                                  textprops=dict(color="w", fontsize= 12))

ax[0].legend(kmeans_cs.index,

          title="Customer Segments",

          loc="center left",

          bbox_to_anchor=(1, 0, 0.5, 1), fontsize= 12)



ax[1].set_title("Proportion of total sales amount", fontsize= 18)

ax[0].set_title("Proportion of Population", fontsize= 18)
plot_dist(kmeans_results, ['Recency', 'First_purchase', 'Frequency', 'Monetary_value',

                           'Unique_items', 'Total_Amount'])