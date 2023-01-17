#Read the CSV files and write to a dataframe



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df_sms=pd.DataFrame()

for i in range (1,8):

    sms=pd.read_csv('../input/mobile-phone-activity/sms-call-internet-mi-2013-11-0{}.csv'.format(i),parse_dates=['datetime'])

    df_sms=df_sms.append(sms)

    
df_sms.head(5)
#get null values by column in dataframe

null_value_counts=df_sms.isnull().sum()

print(null_value_counts)


#Replace nulls with zero for the sms , call and internet columns

df_sms = df_sms.replace(np.nan, 0)



# Generate the totals for sms, call and sms+call+internet

df_sms['sms_total']=df_sms['smsin']+df_sms['smsout']

df_sms['call_total']=df_sms['callin']+df_sms['callout']

df_sms['total_activity']=df_sms['sms_total']+df_sms['call_total']+df_sms['internet']



#  Get day of week and hour for each date time for data analysis 

import datetime

df_sms['datetime'] = pd.to_datetime(df_sms['datetime'])

df_sms['Day'] = df_sms['datetime'].dt.dayofweek

df_sms['hour'] = df_sms['datetime'].dt.hour
df_sms.head(5)
#function  to generate bar plot for activity vs hour and activity vs day

import matplotlib.pyplot as plt

def graph_activity(bytime,activity):

            df_activity=df_sms[activity].groupby(df_sms[bytime]).sum()

            df_activity=df_activity.to_frame()

            fig = plt.figure()

            ax = fig.add_axes([0,0,1,1])

            ax.set_xlabel(bytime)

            ax.set_ylabel(activity)

            ax.set_title('{} by {} '.format(bytime,activity))

            ax.bar(df_activity.index.values,df_activity[activity])

            plt.show()
#generate plot for activity vs hour and activity vs day

bytime=['hour','Day']

activity=['total_activity','sms_total','internet','call_total']

for x in range(0, len(bytime)):

    for y in range(0, len(activity)):

        graph_activity(bytime[x],activity[y])
#GBar graph function to plot cell ids with maximum activity and cell ids wth minimum activity

def bar_by_cell_id(cellid,activity):

    fig,ax = plt.subplots(1,2,figsize=(20,7))

    ax[0].set_xlabel('Cell ID')

    ax[0].set_ylabel(activity)

    ax[0].set_title('Top 10 cell id  by {}'.format(activity))

    ax[0].bar(top_ten_by_activity['CellID'],top_ten_by_activity[activity])

    ax[1].set_xlabel('Cell ID')

    ax[1].set_ylabel(activity)

    ax[1].set_title('Last 10 cell id  by {}'.format(activity))

    ax[1].bar(last_ten_by_activity['CellID'],last_ten_by_activity[activity])

    plt.show()


for i in range(0,len(activity)):

    df_act_cellid=df_sms[activity[i]].groupby(df_sms['CellID']).sum()

    df_act_cellid=df_act_cellid.to_frame()

    top_ten_by_activity=df_act_cellid.nlargest(10,[activity[i]])

    top_ten_by_activity.reset_index(inplace=True)

    top_ten_by_activity['CellID']=top_ten_by_activity['CellID'].astype(str)

    last_ten_by_activity=df_act_cellid.nlargest(10,[activity[i]])

    last_ten_by_activity.reset_index(inplace=True)

    last_ten_by_activity['CellID']=last_ten_by_activity['CellID'].astype(str)

    bar_by_cell_id('CellID',activity[i])



# Group the numerical values by hour to get the hourly totals for each activity



df_group_by_hour=df_sms[['smsin','smsout','callin','callout','internet','sms_total','call_total',

                         'total_activity']].groupby(df_sms['hour']).sum()

df_group_by_hour.reset_index(inplace=True)

df_group_by_hour.head(5)

# Group the numerical values by cell id to get  totals for each activity per cell id

df_group_by_cell_id=df_sms[['smsin','smsout','callin','callout','internet','sms_total','call_total',

                         'total_activity']].groupby(df_sms['CellID']).sum()

df_group_by_cell_id.reset_index(inplace=True)



df_group_by_cell_id.head(5)
# Elbow method to find number of clusters for cell id vs activity



from sklearn.cluster import KMeans

def find_clusters(act):

    wcss = []

    X = df_group_by_cell_id.iloc[:,[0,act]].values

    for i in range(1, 11):

        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)

        kmeans.fit(X)

        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 11), wcss)

    plt.title('The Elbow Method- {}'.format(df_group_by_cell_id.columns[act]))

    plt.xlabel('Number of clusters')

    plt.ylabel('WCSS')

    plt.show()
for act in range(6,9):

    find_clusters(act)
# Implement k means for 4 clusters for cell id vs activity

def impl_KMeans(act):

    X = df_group_by_cell_id.iloc[:,[0,act]].values

    kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)

    y_kmeans = kmeans.fit_predict(X)

    plt.figure(figsize=(20, 10), dpi=80)

    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

    centers = kmeans.cluster_centers_

    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

    plt.title('Clusters of activity by {}'.format(df_group_by_cell_id.columns[act]))

    plt.xlabel('CellID ')

    plt.ylabel('{}'.format(df_group_by_cell_id.columns[act]))

for act in range(6,9):

    impl_KMeans(act)
# Elbow method for hourly vs activity

from sklearn.cluster import KMeans

def find_clusters_hour(act):

    wcss = []

    X = df_group_by_hour.iloc[:,[0,act]].values

    for i in range(1, 11):

        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)

        kmeans.fit(X)

        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 11), wcss)

    plt.title('The Elbow Method- {}'.format(df_group_by_cell_id.columns[act]))

    plt.xlabel('Number of clusters')

    plt.ylabel('WCSS')

    plt.show()
for act in range(6,9):

    find_clusters_hour(act)
# Implement k Means for 3 clusters for hourly vs activity



def impl_KMeans_hour(act):

    X = df_group_by_hour.iloc[:,[0,act]].values

    kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)

    y_kmeans = kmeans.fit_predict(X)

    plt.figure(figsize=(10, 5), dpi=80)

    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

    centers = kmeans.cluster_centers_

    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

    plt.title('Clusters of activity by {}'.format(df_group_by_hour.columns[act]))

    plt.xlabel('Hour ')

    plt.ylabel('{}'.format(df_group_by_hour.columns[act]))

for act in range(6,9):

    impl_KMeans_hour(act)