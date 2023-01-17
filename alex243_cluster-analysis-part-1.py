# import libraries
import pandas as pd
import numpy as np
# import scipy.optimize
# import offer information to df
df_offer = pd.read_csv("../input/OfferInformation.csv")

# rename columns for ease of use
df_offer.columns = ['offer_num', 'campaign', 'varietal', 'min_quantity', 'discount', 'origin', 'past_peak']
# df_offer = df_offer.set_index('offer_num')
df_offer.head()
# import transactions information to df
df_trans = pd.read_csv("../input/Transactions.csv")

# rename columns for ease of use
df_trans.columns = ['name', 'offer_num']
# df_trans = df_trans.set_index('name')
df_trans.head()
pivot_trans = df_trans.pivot_table(index='offer_num', columns='name', aggfunc=len, fill_value=0)
pivot_trans
# convert to df
df_pivot_trans = pd.DataFrame(pivot_trans)

# join 2 dfs
df = df_offer.join(df_pivot_trans, on='offer_num')
df = df.set_index('offer_num')
df
# Insert 4 columns for 4 clusters with initialize value equal to 0
df['cluster1'] = 0
df['cluster2'] = 0
df['cluster3'] = 0
df['cluster4'] = 0
df
np.sqrt(((df.Adams - df.cluster1)**2).sum())
# create an array with name of 4 columns as distance values to each cluster
columns = ['distance_to_cluster1', 'distance_to_cluster2', 'distance_to_cluster3', 'distance_to_cluster4']

# get all customer names
names = df_pivot_trans.columns

# create new dataframe
df_dist_to_cluster = pd.DataFrame(columns=columns, index=names)

# convert columns to float type
for column in columns:
    df_dist_to_cluster[column] = df_dist_to_cluster[column].astype(float)

# apply the formula above for all customers to each cluster
for i in range(len(columns)):
    for index, row in df_dist_to_cluster.iterrows():
        cluster_col = 'cluster' + str(i+1)
        row[columns[i]] = np.sqrt(((df[index] - df[cluster_col])**2).sum())
# find min value
df_dist_to_cluster['min_cluster_distance'] = df_dist_to_cluster.min(axis=1)
df_dist_to_cluster.head()
df_dist_to_cluster['assigned_cluster'] = 0
for index, row in df_dist_to_cluster.iterrows():
    for i in range(len(columns)):
        if row[columns[i]] == row['min_cluster_distance']:
            df_dist_to_cluster.loc[index, 'assigned_cluster'] = i + 1
            break
            
df_dist_to_cluster.head()
# import cluster points from csv file
df_clusters = pd.read_csv('../input/Clusters.csv')
df_clusters = df_clusters.set_index(df.index)
# update df
df[['cluster1', 'cluster2', 'cluster3', 'cluster4']] = df_clusters[['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']]
df.head()

# Recalculate distance
for index, row in df_dist_to_cluster.iterrows():
    for i in range(len(columns)):
        cluster_col = 'cluster' + str(i+1)
        df_dist_to_cluster.loc[index, columns[i]] = np.sqrt(((df[index] - df[cluster_col])**2).sum())
#         print(row[columns[i]], columns[i])
        
# find min value
df_dist_to_cluster['min_cluster_distance'] = df_dist_to_cluster[columns].min(axis=1)
df_dist_to_cluster.head()
for index, row in df_dist_to_cluster.iterrows():
    for i in range(len(columns)):
        if row[columns[i]] == row['min_cluster_distance']:
            df_dist_to_cluster.loc[index, 'assigned_cluster'] = i + 1
            break
            
df_dist_to_cluster
group1_name = df_dist_to_cluster.index[df_dist_to_cluster['assigned_cluster'] == 1]

group1_list = []
for name in group1_name:
    group1_list.append(df.index[df[name] == 1].values)
group1_deal_number = []
for sub_list in group1_list:
    for item in sub_list:
        group1_deal_number.append(item)
# group1_deal_number = list(set(group1_deal_number))
# len(group1_deal_number)
df.loc[group1_deal_number]
group2_name = df_dist_to_cluster.index[df_dist_to_cluster['assigned_cluster'] == 2]
group2_name
group2_list = []
for name in group2_name:
    group2_list.append(df.index[df[name] == 1].values)
group2_deal_number = []
for sub_list in group2_list:
    for item in sub_list:
        group2_deal_number.append(item)
df.loc[group2_deal_number]
df_top_deal_by_cluster = pd.DataFrame(df, index=df.index, 
                                      columns=['campaign', 'varietal', 'min_quantity', 'discount', 
                                               'origin', 'past_peak'])
# get customer names of group 3, 4
group3_name = df_dist_to_cluster.index[df_dist_to_cluster['assigned_cluster'] == 3]
group4_name = df_dist_to_cluster.index[df_dist_to_cluster['assigned_cluster'] == 4]

# create pivot table set index equal to name so we can join with distance to cluster dataframe
pivot_trans_vertical = df_trans.pivot_table(index='name', columns='offer_num', aggfunc=len, fill_value=0)
df_pivot_trans_vertical = pd.DataFrame(pivot_trans_vertical)

# join with distance to cluster
df_groups = df_dist_to_cluster.join(df_pivot_trans_vertical)
df_groups.loc[group1_name]

# find number of each deal for each group
df_top_deal_by_cluster['G1'] = df_groups.loc[group1_name][df_top_deal_by_cluster.index].sum()
df_top_deal_by_cluster['G2'] = df_groups.loc[group2_name][df_top_deal_by_cluster.index].sum()
df_top_deal_by_cluster['G3'] = df_groups.loc[group3_name][df_top_deal_by_cluster.index].sum()
df_top_deal_by_cluster['G4'] = df_groups.loc[group4_name][df_top_deal_by_cluster.index].sum()
df_top_deal_by_cluster
# group 1
df_top_deal_by_cluster.sort_values(by=['G1'], ascending=False).head(5)
df_top_deal_by_cluster.sort_values(by=['G2'], ascending=False).head(5)
df_top_deal_by_cluster.sort_values(by=['G3'], ascending=False).head(10)
df_top_deal_by_cluster.sort_values(by=['G4'], ascending=False).head(10)
