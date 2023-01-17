!pip install kneed
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import time

import datetime as datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from kneed import KneeLocator

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
cf.set_config_file(offline=True)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path_file = r'/kaggle/input/smart-meter-dataset/'
def MinMaxScaler(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))

def Kmeans_clustering(df, clusterNum, max_iter, n_jobs):
    scaler = StandardScaler()
    scaler.fit(df)
    df_std = pd.DataFrame(data=scaler.transform(df), columns=df.columns, index=df.index)
    km_model = KMeans(n_clusters=clusterNum, max_iter=max_iter, random_state=666)
    km_model = km_model.fit(df_std)
    clusterdf= pd.DataFrame(data=km_model.labels_, columns=['ClusterNo'])
    clusterdf.index = df.index
    return clusterdf

def Kmeans_bestClusterNum(df, range_min, range_max, max_iter, n_jobs):
    silhouette_avgs = []
    sum_of_squared_distances = []
    
    ks = range(range_min,range_max+1)
    for k in ks:
        kmeans_fit = KMeans(n_clusters = k, max_iter=max_iter, random_state=666).fit(df)
        cluster_labels = kmeans_fit.labels_
        sum_of_squared_distances.append(kmeans_fit.inertia_)
        
    kn = KneeLocator(list(ks), sum_of_squared_distances, S=1.0, curve='convex', direction='decreasing')  
    plt.xlabel('k')
    plt.ylabel('sum_of_squared_distances')
    plt.title('The Elbow Method showing the optimal k')
    plt.plot(ks, sum_of_squared_distances, 'bx-')
    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.show()
    print('Optimal clustering number:'+str(kn.knee))
    print('----------------------------')    
    
    return kn.knee
df_metadata_loop = pd.read_csv(os.path.join(path_file, 'metadata_loop.csv'))
df_metadata_loop
df_metadata_uid = pd.read_csv(os.path.join(path_file, 'metadata_uid.csv'))
df_metadata_uid
df_powerMeter_pivot_output = pd.read_csv(os.path.join(path_file, 'powerMeter.csv'))
df_powerMeter_pivot_output['日期時間'] = pd.to_datetime(df_powerMeter_pivot_output['日期時間'])
df_powerMeter_pivot_output = df_powerMeter_pivot_output.set_index('日期時間')
df_powerMeter_pivot_output
# Leave meters with building type of '行政單位'
building_type = '行政單位'

df_metadata = df_metadata_loop.merge(df_metadata_uid, on='uid')
list_powerMeter = df_metadata[df_metadata['buildType1C']==building_type]['迴路編號'].to_list()
df_powerMeter_pivot_output = df_powerMeter_pivot_output.loc[:, df_powerMeter_pivot_output.columns.str.contains('|'.join(list_powerMeter))]
df_powerMeter_pivot_output.columns
df_PM_temp = df_powerMeter_pivot_output.copy()
df_PM_temp = (df_PM_temp-df_PM_temp.mean())/df_PM_temp.std()
df_PM_temp = df_PM_temp.T

try:
    bestClusterNum_dept = Kmeans_bestClusterNum(df=df_PM_temp.fillna(0), range_min=2, range_max=20, max_iter=10000, n_jobs=-1)
except:
    try:
        bestClusterNum_dept = Kmeans_bestClusterNum(df=df_PM_temp.fillna(0), range_min=2, range_max=15, max_iter=10000, n_jobs=-1)    
    except:
        try:
            bestClusterNum_dept = Kmeans_bestClusterNum(df=df_PM_temp.fillna(0), range_min=2, range_max=10, max_iter=10000, n_jobs=-1)    
        except:
            bestClusterNum_dept = 3

try:
    df_PM_temp['ClusterNo'] = Kmeans_clustering(df=df_PM_temp.fillna(0), clusterNum=bestClusterNum_dept, max_iter=100000, n_jobs=-1)
except:
    bestClusterNum_dept = 3
    df_PM_temp['ClusterNo'] = Kmeans_clustering(df=df_PM_temp.fillna(0), clusterNum=bestClusterNum_dept, max_iter=100000, n_jobs=-1)

for ClusterNo in df_PM_temp['ClusterNo'].sort_values().unique():
    df_plot = df_PM_temp[df_PM_temp['ClusterNo']==ClusterNo].T.drop('ClusterNo')
    print('ClusterNo: ' + str(ClusterNo))    
    print('Amount of meters: ' + str(len(df_plot.T)))
    df_plot.plot(figsize=(15,5),color='black',alpha=0.1,legend=False, ylim=(-5, 10))
    plt.xlabel("")
    plt.show()
    print('---------------------------------------------------------------------------------------------------')
df_cluster = df_PM_temp.iloc[:,-1].reset_index().copy()
for idx_1 in df_cluster.index:
    for idx_2 in df_metadata_loop.index:
        name_loop = df_metadata_loop.loc[idx_2, '迴路編號']
        if name_loop in df_cluster.loc[idx_1, 'index']:
            df_cluster.loc[idx_1, '迴路編號'] = name_loop
df_cluster
df_cluster = df_cluster.merge(df_metadata_loop, on='迴路編號')
df_cluster = df_cluster.merge(df_metadata_uid, on='uid')
df_cluster
df_plot = df_cluster.pivot_table(columns='buildType1C', index='ClusterNo', values='index', aggfunc='count')
df_plot.plot.bar(stacked=True, figsize=(15,5), title='Number of meters in each cluster', legend=False)
df_energy = df_powerMeter_pivot_output.mean() * df_powerMeter_pivot_output.count().max()
df_energy = df_energy.reset_index().rename(columns={0:'energy_sum'})
df_energy = df_energy.merge(df_cluster, on='index')
df_energy['EUI'] = df_energy['energy_sum'] / df_energy['area'].replace(',','.', regex=True).astype('float')
df_energy
plt.figure(figsize=(3,5))
ax = sns.boxplot(x="buildType1E", y="EUI", data=df_energy[df_energy['EUI']<10**3].sort_values('buildType1E'))
ax = sns.swarmplot(x="buildType1E", y="EUI", data=df_energy[df_energy['EUI']<10**3].sort_values('buildType1E'), color=".25")
plt.figure(figsize=(10,5))
ax = sns.boxplot(x="ClusterNo", y="EUI", data=df_energy[df_energy['EUI']<10**3])
ax = sns.swarmplot(x="ClusterNo", y="EUI", data=df_energy[df_energy['EUI']<10**3], color=".25")


