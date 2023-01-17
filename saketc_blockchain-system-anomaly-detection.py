from google.cloud import bigquery

from scipy.stats.mstats import zscore

from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

from sklearn.cluster import KMeans

from sklearn.manifold import TSNE

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import matplotlib as mpl

from pathlib import Path

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import IsolationForest 

import seaborn as sns

import datetime as dt

from datetime import datetime,tzinfo

import scipy, json, csv, time, pytz

from pytz import timezone

import numpy as np

import pandas as pd

seed = 135

%config InlineBackend.figure_format = 'retina'

%matplotlib inline

import os

os.listdir('../input/gcp-bitcoin-project/')
#Connecting to Google datastore (use path to ur private key)

os.environ['GOOGLE_APPLICATION_CREDENTIALS']="../input/gcp-bitcoin-project/Bitcoin Project-615d07137267.json"

client = bigquery.Client()
# The query to get date, number of transactions from Google BigQuery bitcoin blockchain dataset 

# Select records from the last three years and group them with respect to date

query_1 = """

SELECT 

   DATE(TIMESTAMP_MILLIS(timestamp)) AS Date,

   COUNT(transactions) AS Transactions

FROM `bigquery-public-data.bitcoin_blockchain.blocks`

GROUP BY date

HAVING date >= '2016-08-12' AND date <= '2019-08-12'

ORDER BY date

"""

query_job_1 = client.query(query_1)

# Waits for the query to finish

iterator_1 = query_job_1.result(timeout=30)

rows_1 = list(iterator_1)

df_1 = pd.DataFrame(data=[list(x.values()) for x in rows_1], columns=list(rows_1[0].keys()))
# The query to get sum of all satoshis spent each day and number of blocks

query_2 = """

SELECT

  o.Date,

  COUNT(o.block) AS Blocks,

  SUM(o.output_price) AS Output_Satoshis

FROM (

  SELECT

    DATE(TIMESTAMP_MILLIS(timestamp)) AS Date,

    output.output_satoshis AS output_price,

    block_id AS block

  FROM

    `bigquery-public-data.bitcoin_blockchain.transactions`,

    UNNEST(outputs) AS output ) AS o

GROUP BY

  o.date

HAVING o.date >= '2016-08-12' AND o.date <= '2019-08-12'

ORDER BY o.date, blocks

"""

query_job_2 = client.query(query_2)

# Waits for the query to finish

iterator_2 = query_job_2.result(timeout=30)

rows_2 = list(iterator_2)

df_2 = pd.DataFrame(data=[list(x.values()) for x in rows_2], columns=list(rows_2[0].keys()))



df_2["Output_Satoshis"]= df_2["Output_Satoshis"].apply(lambda x: float(x/100000000))
df_1.head()
df_2.head()
# merge the two dataframes

result = pd.merge(df_1,

                 df_2[['Date', 'Blocks', 'Output_Satoshis']],

                 on='Date')

result.head()
# Number of records 

len(result)
# get the overview of our data

result.describe()
sns.kdeplot(result['Blocks'])
sns.kdeplot(result['Transactions'])
sns.kdeplot(result['Output_Satoshis'])
%matplotlib inline

plt.style.use('ggplot')

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 1.5})



g = plt.subplots(figsize=(20,9))

g = sns.lineplot(x='Date', y='Transactions', data=result, palette='Blues_d')

plt.title('Transactions per day')
%matplotlib inline

plt.style.use('ggplot')

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 1.5})



g = plt.subplots(figsize=(20,9))

g = sns.lineplot(x='Date', y='Blocks', data=result, palette='Blues_d')

plt.title('Blocks per day')
g = plt.subplots(figsize=(20,9))

g = sns.lineplot(x='Date', y='Output_Satoshis', data=result, palette='BuGn_r')

plt.title('Sum of all satoshis spent each day')
# check the relation among the features of data

sns.set(style="ticks")

sns.pairplot(result)
# select the three most important features (Transactions, Blocks, Output Satoshis) from the data

data = result[['Output_Satoshis','Blocks','Transactions']]

outliers_fraction=0.05

scaler = StandardScaler()

np_scaled = scaler.fit_transform(data)

data = pd.DataFrame(np_scaled)



# train isolation forest

model =  IsolationForest(contamination=outliers_fraction)

model.fit(data) 
fig = plt.figure(figsize=(10,6))

ax = fig.add_subplot(111,projection='3d')

X = result.iloc[:,1:4].values

colors = np.array(['red', 'blue'])

y_pred = model.fit_predict(data)

ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=25, color=colors[(y_pred + 1) // 2] )

ax.legend()

#plt.xlabel('Transactions')

#plt.ylabel('Blocks')

#plt.zlabel('Sum of Output Satoshis')

plt.title('Transactions vs Blocks vs Sum of Output Satoshis: Red represents Anomalies')

plt.savefig('IsolationForest_anomaly.png', dpi=1000)
# create a new column for storing the results of Isolation Forest method

result['anomaly_IsolationForest'] = pd.Series(model.predict(data))

result['anomaly_IsolationForest'] = result['anomaly_IsolationForest'].apply(lambda x: x == -1)

result['anomaly_IsolationForest'] = result['anomaly_IsolationForest'].astype(int)

result['anomaly_IsolationForest'].value_counts()
fig, ax = plt.subplots(figsize=(10,6))



#anomaly

a = result.loc[result['anomaly_IsolationForest'] == 1]

ax.plot(result['Transactions'], color='black', label = 'Normal', linewidth=1.5)

ax.scatter(a.index ,a['Transactions'], color='red', label = 'Anomaly', s=16)

plt.legend()

plt.title("Anamoly Detection Using Isolation Forest")

plt.xlabel('Date')

plt.ylabel('Transactions')

plt.savefig('IsolationForest_anomaly_Transactions.png', dpi=1000)

plt.show();
fig, ax = plt.subplots(figsize=(10,6))



#anomaly

a = result.loc[result['anomaly_IsolationForest'] == 1]

ax.plot(result['Blocks'], color='black', label = 'Normal', linewidth=1.5)

ax.scatter(a.index ,a['Blocks'], color='red', label = 'Anomaly', s=16)

plt.legend()

plt.title("Anamoly Detection Using Isolation Forest")

plt.xlabel('Date')

plt.ylabel('Blocks')

plt.savefig('IsolationForest_anomaly_Blocks.png', dpi=1000)

plt.show();
fig, ax = plt.subplots(figsize=(10,6))



#anomaly

a = result.loc[result['anomaly_IsolationForest'] == 1]

ax.plot(result['Output_Satoshis'], color='black', label = 'Normal', linewidth=1.5)

ax.scatter(a.index ,a['Output_Satoshis'], color='red', label = 'Anomaly', s=16)

plt.legend()

plt.title("Anamoly Detection Using Isolation Forest")

plt.xlabel('Date')

plt.ylabel('Sum of Output Satoshis')

plt.savefig('IsolationForest_anomaly_Output_Satoshis.png', dpi=1000)

plt.show();
# This code has been taken from kernel https://github.com/anish-saha/EventDetection-Paradigm01/blob/master/KMeans.ipynb



def pca_results(good_data, pca):

    # Dimension indexing

    dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]



    # PCA components

    components = pd.DataFrame(np.round(pca.components_, 4), columns = good_data.keys())

    components.index = dimensions



    # PCA explained variance

    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)

    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])

    variance_ratios.index = dimensions



    # Create a bar plot visualization

    fig, ax = plt.subplots(figsize = (10,10))



    # Plot the feature weights as a function of the components

    components.plot(ax = ax, kind = 'bar');

    ax.set_ylabel("Feature Weights")

    ax.set_xticklabels(dimensions, rotation=0)





    # Display the explained variance ratios

    for i, ev in enumerate(pca.explained_variance_ratio_):

        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n %.4f"%(ev))



    # Return a concatenated DataFrame

    return pd.concat([variance_ratios, components], axis = 1)
data_ = data.copy() # make a copy of data with three already selected features

data_ = data_.reset_index(drop=True)



data_[:] = MinMaxScaler().fit_transform(data_[:])

pca = PCA(n_components=2) # we have selected 2 components in PCA for simplicity

pca.fit(data_)

reduced_data = pca.transform(data_)

reduced_data = pd.DataFrame(reduced_data)



num_clusters = range(1, 20)



kmeans = [KMeans(n_clusters=i, random_state=seed).fit(reduced_data) for i in num_clusters]

scores = [kmeans[i].score(reduced_data) for i in range(len(kmeans))]



fig, ax = plt.subplots(figsize=(8,6))

ax.plot(num_clusters, scores, linewidth = 4)

plt.xticks(num_clusters)

plt.xlabel('Number of Clusters')

plt.ylabel('Score')

plt.title('Elbow Curve')

plt.show();
correlations = pd.DataFrame(data=data_).corr()

pca_results(correlations, pca)
#Choosing the three clusters based on the elbow curve

best_num_cluster__ = 3

km__ = KMeans(n_clusters=best_num_cluster__, random_state=seed)

km__.fit(reduced_data)

km__.predict(reduced_data)

labels__1 = km__.labels_



#Choosing the four clusters based on the elbow curve

best_num_cluster = 4

km = KMeans(n_clusters=best_num_cluster, random_state=seed)

km.fit(reduced_data)

km.predict(reduced_data)

labels = km.labels_



#Choosing the five clusters based on the elbow curve

best_num_cluster_ = 5

km_ = KMeans(n_clusters=best_num_cluster_, random_state=seed)

km_.fit(reduced_data)

km_.predict(reduced_data)

labels_1 = km_.labels_

#Plotting based on three cluster

fig = plt.figure(1, figsize=(7,7))

plt.scatter(reduced_data.iloc[:,0], reduced_data.iloc[:,1], 

            c=labels__1.astype(np.float), edgecolor="k", s=16)

plt.xlabel('Principal Component 1')

plt.ylabel('Principal Component 2')

plt.title('Clusters based on K means: 3 clusters')
#Plotting based on four cluster

fig = plt.figure(1, figsize=(7,7))

plt.scatter(reduced_data.iloc[:,0], reduced_data.iloc[:,1], 

            c=labels.astype(np.float), edgecolor="k", s=16)

plt.xlabel('Principal Component 1')

plt.ylabel('Principal Component 2')

plt.title('Clusters based on K means: 4 clusters')
#Plotting based on five cluster

fig = plt.figure(1, figsize=(7,7))

plt.scatter(reduced_data.iloc[:,0], reduced_data.iloc[:,1], 

            c=labels_1.astype(np.float), edgecolor="k", s=16)

plt.xlabel('Principal Component 1')

plt.ylabel('Principal Component 2')

plt.title('Clusters based on K means: 5 clusters')
reduced_data.loc[0]

mod = kmeans[best_num_cluster-1]

mod.cluster_centers_
reduced_data['Principal Component 1'] = reduced_data[0]

reduced_data['Principal Component 2'] = reduced_data[1]

reduced_data.drop(columns = [0, 1], inplace=True)

reduced_data.head()
def getDistanceByPoint(data, model):

    distance = []

    for i in range(0,len(data)):

        Xa = np.array(data.loc[i])

        Xb = model.cluster_centers_[model.labels_[i]-1]

        distance.append(np.linalg.norm(Xa-Xb))

    return distance



outliers_fraction = 0.05

# find the distance between each point and its nearest centroid. The largest distances will be consdiered anomalies

distance = getDistanceByPoint(reduced_data, kmeans[best_num_cluster-1])

distance = pd.Series(distance)

number_of_outliers = int(outliers_fraction*len(distance))

threshold = distance.nlargest(number_of_outliers).min()





# anomaly_kmeans contain the anomaly result of the above method  (0:normal, 1:anomaly) 

result['anomaly_kmeans'] = (distance >= threshold).astype(int)



# visualisation of anomaly with cluster view

#fig, ax = plt.subplots(figsize=(10,6))

colors = {0:'blue', 1:'red'}

#colors = {1:'#f70505', 0:'#0a48f5'}

plt.figure(figsize=(7,7))

plt.scatter(reduced_data.iloc[:,0], reduced_data.iloc[:,1], 

            c=result["anomaly_kmeans"].apply(lambda x: colors[x]), s=25)

plt.xlabel('principal feature 1')

plt.ylabel('principal feature 2')

plt.title('Anomaly prediction using KMeans: Red represents Anomaly')

plt.savefig('KMeans_anomaly.png', dpi=1000)
result['anomaly_kmeans'].value_counts()
fig, ax = plt.subplots(figsize=(10,6))



#anomaly

a = result.loc[result['anomaly_kmeans'] == 1]

ax.plot(result['Transactions'], color='black', label = 'Normal', linewidth=1.5)

ax.scatter(a.index ,a['Transactions'], color='red', label = 'Anomaly', s=16)

plt.legend()

plt.title("Anamoly Detection Using Kmeans")

plt.xlabel('Date')

plt.ylabel('Transactions')

plt.savefig('KMeans_anomaly_Transactions.png', dpi=1000)

plt.show();
fig, ax = plt.subplots(figsize=(10,6))



#anomaly

a = result.loc[result['anomaly_kmeans'] == 1]

ax.plot(result['Blocks'], color='black', label = 'Normal', linewidth=1.5)

ax.scatter(a.index ,a['Blocks'], color='red', label = 'Anomaly', s=16)

plt.legend()

plt.title("Anamoly Detection Using Kmeans")

plt.xlabel('Date')

plt.ylabel('Blocks')

plt.savefig('KMeans_anomaly_Blocks.png', dpi=1000)

plt.show();
fig, ax = plt.subplots(figsize=(10,6))



#anomaly

a = result.loc[result['anomaly_kmeans'] == 1]

ax.plot(result['Output_Satoshis'], color='black', label = 'Normal', linewidth=1.5)

ax.scatter(a.index ,a['Output_Satoshis'], color='red', label = 'Anomaly', s=16)

plt.legend()

plt.title("Anamoly Detection Using Kmeans")

plt.xlabel('Date')

plt.ylabel('Sum of Output Satoshis')

plt.savefig('KMeans_anomaly_Output_Satoshis.png', dpi=1000)

plt.show();
# final result dataframe

result.head()
# select the cases for final anomaly in which both the algorithms predicted anomaly

final_anomaly = result.query('anomaly_kmeans == 1 & anomaly_IsolationForest == 1')

final_anomaly.head()
# Select the cases in which either of the two algorithms predicted anomaly

possible_anomaly = result.query('anomaly_kmeans == 1 | anomaly_IsolationForest == 1')

possible_anomaly.head()
# Select the cases where no algorithm predicted anomaly

no_anomaly = result.query('anomaly_kmeans == 0 & anomaly_IsolationForest == 0')

no_anomaly.head()
total_anomaly = len(final_anomaly)+len(possible_anomaly)

percent_total_anomaly = total_anomaly*100/len(result)

print('Total records:',len(result))

print('Number of final anomaly:', len(final_anomaly))

print('Number of possible anomaly:', len(possible_anomaly))

print('Total anomaly:', total_anomaly)

print('Percentage of total anomaly in the data: %0.2f' % percent_total_anomaly)