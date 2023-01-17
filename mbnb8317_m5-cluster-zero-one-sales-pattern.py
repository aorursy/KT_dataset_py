import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os



import matplotlib.pyplot as plt

import seaborn as sns



from tqdm import tqdm

from scipy.cluster.hierarchy import linkage, fcluster



import warnings

warnings.filterwarnings("ignore")
grid_df = pd.read_pickle('/kaggle/input/m5-simple-fe/grid_part_1.pkl')
grid_df.head()
grid_df = grid_df[['id','d','sales']].pivot(index='id',columns='d').reset_index()

ids = pd.DataFrame(grid_df['id'])
grid_df = grid_df['sales'].iloc[:,:1913]



# values > 0 for 1, missing for nan, 0 for -1

grid_df = pd.DataFrame(np.where(grid_df.isnull(),np.nan,

                                np.where(grid_df > 0, 1, -1)))



grid_df.columns = [f'd_{i}' for i in range(1,1914)]
grid_df.head()
d1_peak = grid_df.notnull().sum(axis=1)

cluster = d1_peak.copy()
plt.figure(figsize=(12,6))

sns.kdeplot(d1_peak)

plt.plot([925,925],[0,0.0030]); plt.plot([1300,1300],[0,0.0030]); plt.plot([1700,1700],[0,0.0030])

plt.text(x=700,y=0.002,s='Cluster 1'); plt.text(x=1020,y=0.002,s='Cluster 2')

plt.text(x=1420,y=0.002,s='Cluster 3'); plt.text(x=2000,y=0.002,s='Cluster 4')

plt.title('# of Nan distribution')

plt.show()
c1_mask = (d1_peak <= 925)

c2_mask = (d1_peak > 925) & (d1_peak <= 1300)

c3_mask = (d1_peak > 1300) & (d1_peak <= 1700)

c4_mask = (d1_peak > 1700)



cluster[c1_mask] = 1

cluster[c2_mask] = 2

cluster[c3_mask] = 3

cluster[c4_mask] = 4
grid_df['cluster'] = cluster



# missing values for 0

grid_df = grid_df.fillna(0)
grid_df['cluster'].value_counts()
A = np.array([1,0,0,0,1,0])

B = np.array([1,1,0,0,0,0])
np.sum(A == B) / len(A)
cluster_df = grid_df[grid_df['cluster'] == 1]

cluster_array = cluster_df.values

cluster_array = np.where(cluster_array == 0, np.nan, cluster_array)
length = cluster_array.shape[0] 

for i in tqdm(range(0, int(length/10))):

    for j in range(i, length):

        np.sum(cluster_array[i,:-1] == cluster_array[j,:-1])
def Clustering(cluster_lv1_name, cluster_lv2_num):

    

    cluster_df = grid_df[grid_df['cluster'] == cluster_lv1_name]

    

    if cluster_lv2_num == 1:

        print('Pass : Cluster', cluster_lv1_name)

        

    else:

        print('Making dist_matrix : Cluster', cluster_lv1_name)

        cluster_array = cluster_df.values

        dist_matrix = np.dot(cluster_array, cluster_array.T)



        ## this part, linkage, takes about 30 minutes.

        ## If you have another idea for reducing running time,

        ## Please advise me !

        Z = linkage(dist_matrix, method='ward')

        cluster_num = fcluster(Z, t=cluster_lv2_num, criterion='maxclust')

        cluster_df['cluster'] = cluster_df['cluster'].astype(str) + '_' + cluster_num.astype(str)



    return cluster_df
plan_clustering = {

    #cluster_lv1_name : how many cluster_lv2 to make

    1:1,

    2:1,

    3:1,

    4:4

}
%time

df_list = list()

for lv1, lv2 in plan_clustering.items():

    df_name = f'cluster_{lv1}_df'



    cluster_df = Clustering(cluster_lv1_name = lv1, cluster_lv2_num = lv2)

    globals()[df_name] = cluster_df

    

    df_list += [cluster_df['cluster']]
cls_total = pd.concat(df_list)
cluster_df = pd.concat([ids, cls_total], axis=1)
cluster_df
cluster_df['cluster'].value_counts()
cluster_df.to_pickle('zero_one_cluster.pkl')