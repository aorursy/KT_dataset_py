# import librairies
import numpy as np
import pandas as pd
import time 
import requests
import json
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = [16,13]
!ls ../input/toronto-bikeshare-data
#will use data q2 q3 q4 and 2017 later
q1 = pd.read_csv('../input/toronto-bikeshare-data/bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q1 2018.csv')
# q2 = pd.read_csv('bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q2 2018.csv')
# q3 = pd.read_csv('bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q3 2018.csv')
# q4 = pd.read_csv('bikeshare2018/bikeshare2018/Bike Share Toronto Ridership_Q4 2018.csv')
#toronto = [q1, q2, q3, q4] 
toronto = [q1]
df = pd.concat(toronto)
df.head()
# get the stations information from https://tor.publicbikesystem.net
req = requests.get('https://tor.publicbikesystem.net/ube/gbfs/v1/en/station_information')
stations = json.loads(req.content)['data']['stations']
stations = pd.DataFrame(stations)[['station_id', 'name', 'lat', 'lon']].astype({
    'station_id': 'float64',
})

stations.head()
from sklearn.cluster import KMeans
X = stations[['lon', 'lat']].values
n_clusters_ = 20
kmeans = KMeans(n_clusters = n_clusters_, init ='k-means++')
kmeans.fit(X) # Compute k-means clustering.
labels = kmeans.fit_predict(X)
stations['cluster'] = labels
#dfc -> df with cluster column
dfc = pd.merge(df
                 , stations[['station_id','cluster']]
                 , how='left', left_on=['from_station_id']
                 , right_on=['station_id']) \
                        .rename(columns={"cluster": "cluster_origin"})
dfc = pd.merge(dfc
                 , stations[['station_id','cluster']]
                 , how='left', left_on=['to_station_id']
                 , right_on=['station_id']) \
                        .rename(columns={"cluster": "cluster_destination"})
dfc = dfc.drop(columns=['station_id_x', 'station_id_y'])
dfc
df_inside_cluster  = dfc[dfc['cluster_origin']==dfc['cluster_destination']] 
df_outside_cluster  = dfc[dfc['cluster_origin']!=dfc['cluster_destination']] 
df_inside_cluster.head()
#df_ic_list ->list df_inside_cluster :: df_clust_0..df_clust_19
df_ic_list = []
for i in range(20):
    exec("df_clust_{} = df_inside_cluster[df_inside_cluster['cluster_destination']=={}]".format(i,i))
    exec("df_ic_list.append(df_clust_{})".format(i))
df_clust_9.head()
## just ignore the warning, dont know why?
for dfitem in df_ic_list:
    if len(dfitem)<100:
        dfitem['score'] = 0
        dfitem['anomaly']=1
        continue
    model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)
    model.fit(dfitem[['trip_duration_seconds']])
    dfitem['score']=model.decision_function(dfitem[['trip_duration_seconds']])
    dfitem['anomaly']=model.predict(dfitem[['trip_duration_seconds']])
df_clust_9[df_clust_9['anomaly']==-1]
df_clust_18.head()
len(df_clust_15.loc[df_clust_15['anomaly']==-1]),len(df_clust_15.loc[df_clust_15['anomaly']==1])
#df_c_list ->list df_outside_cluster :: df_oc_1to1..df_oc_19to19
df_oc_list = []
dfoc = []
for i in range(20):
    for j in range(20):
        if i!=j:
            exec("df_oc_{}to{}=df_outside_cluster[(df_outside_cluster['cluster_origin'] == {})&(df_outside_cluster['cluster_destination']=={})]".format(i,j,i,j))
            exec("df_oc_list.append(df_oc_{}to{})".format(i,j))
## just ignore the warning, dont know why?
for dfitem in df_oc_list:
    if len(dfitem)<100:
        dfitem['score'] = 0
        dfitem['anomaly']=1
        continue
    model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)
    model.fit(dfitem[['trip_duration_seconds']])
    dfitem['score']=model.decision_function(dfitem[['trip_duration_seconds']])
    dfitem['anomaly']=model.predict(dfitem[['trip_duration_seconds']])
   
df_oc_0to15.head()
len(df_oc_0to15.loc[df_oc_0to15['anomaly']==-1]),len(df_oc_0to15.loc[df_oc_0to15['anomaly']==1])
dfc
#listFinal -> lets unite all the df :: df_df_clust_0..df_clust_19 ++  df_oc_0to1..df_oc_18to19
listFinal = []
for i in range(20):
    exec("listFinal.append(df_clust_{})".format(i))

for i in range(20):
    for j in range(20):
        if i!=j:
            exec("listFinal.append(df_oc_{}to{})".format(i,j))
        
#left join to the dfc 
for df_item in listFinal: 
    dfc = dfc.join(df_item['anomaly'], lsuffix='_1', rsuffix='_2')

dfc.head()
#sparse NaN, lets collapse them
anomali = dfc.filter(like='anomaly')
anomali = anomali.fillna(0).sum(axis=1).to_frame()
anomali


#filter column with str 'anomaly'
dfc = dfc.drop(list(dfc.filter(regex='anomaly')),axis=1)
dfc.head()
dfc['anomaly']=anomali
dfc.head()
#count each label
dfc['anomaly'].value_counts()
17294/len(dfc)*100
df_from = df['from_station_id'].value_counts().reset_index()
df_from.columns = ['id','count_from']
df_to = df['to_station_id'].value_counts().reset_index()
df_to.columns = ['id','count_to']
df_to.head()
df_from.head()
df_round = pd.merge(df_from,df_to,left_on='id',right_on='id')
df_round['diff'] = df_round['count_from'].sub(df_round['count_to'])
df_round['percent']=df_round['count_to']/df_round['count_from']*100
df_round.head()
df_round['percent'].mean()
