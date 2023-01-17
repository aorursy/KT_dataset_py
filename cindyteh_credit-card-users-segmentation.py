import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# read data into dataframe
df =  pd.read_csv('../input/german-credit/german_credit_data.csv', index_col=0)
df.head(2)
# understand data type
df.dtypes
# check for null values
df.isnull().sum()
print('Content of saving accounts:', df['Saving accounts'].unique())
print('Content of checking account:', df['Checking account'].unique())
df['Saving accounts'] = df['Saving accounts'].fillna('none')
df['Checking account'] = df['Checking account'].fillna('none')
df.head(2)
sns.set_style('white') 
fig, ax = plt.subplots(3,2,figsize=(16,15))
sns.countplot(df['Job'], ax=ax[0][0], palette=sns.cubehelix_palette())
sns.countplot(df['Housing'], ax=ax[0][1], palette=sns.color_palette('BrBG'))
sns.countplot(df['Saving accounts'], ax=ax[1][0], palette=sns.color_palette('BuGn_r'))
sns.countplot(df['Checking account'], ax=ax[1][1],palette=sns.color_palette('RdBu_r')[4:])
sns.countplot(df['Purpose'], ax=ax[2][0], palette=sns.color_palette('Paired'))
sns.countplot(df['Sex'], ax=ax[2][1],palette=sns.light_palette('navy')[3:])

ax[2][0].tick_params(labelrotation=45)
fig, ax = plt.subplots(3,1,figsize=(10,13))
plt.tight_layout(4)

sns.lineplot(data=df, x='Age', y='Credit amount', hue='Sex', lw=2, ax=ax[0], palette='Set2')
sns.lineplot(data=df, x='Age', y='Duration', hue='Sex', lw=2, ax=ax[1], palette='Set2')
sns.lineplot(data=df, x='Duration', y='Credit amount', hue='Sex', lw=2, ax=ax[2],  palette='Set2')
fig = plt.subplots(figsize=(10,7))
data = df[['Age','Credit amount', 'Duration']]
sns.heatmap(data.corr(), annot = True, cmap='BuPu')
# Data for K-means clustering
data = df[['Age', 'Credit amount', 'Duration']]
data.head()
# Distribution of Age, Credit amount and Duration
fig, ax = plt.subplots(3,1,figsize=(10,9))
plt.tight_layout(4)

sns.distplot(data['Age'], color='mediumvioletred', bins=80, hist=False, ax=ax[0])
sns.distplot(data['Credit amount'], color='mediumvioletred', bins=80, hist=False, ax=ax[1])
sns.distplot(data['Duration'], color='mediumvioletred', bins=80, hist=False, ax=ax[2])
data_log = np.log(data)
data_log.head()
fig, ax = plt.subplots(3,1,figsize=(10,9))
plt.tight_layout(4)

sns.distplot(data_log['Age'], color='mediumvioletred', bins=80, hist=False, ax=ax[0])
sns.distplot(data_log['Credit amount'], color='mediumvioletred', bins=80, hist=False, ax=ax[1])
sns.distplot(data_log['Duration'], color='mediumvioletred', bins=80, hist=False, ax=ax[2])
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_log)
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(data_scaled)
    distortions.append(kmeanModel.inertia_)
plt.figure(figsize=(10,5))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
# k-means algorithm
k = 4
kmeans = KMeans(n_clusters=k, random_state=0).fit(data_scaled)
df['Cluster'] = kmeans.labels_
df['Cluster'] = df['Cluster'].astype('category')
fig = px.scatter_3d(df, x='Age', y='Duration', z='Credit amount',
              color='Cluster')
fig.show()
cluster_size = df.groupby(['Cluster'], as_index=False).size()
cluster_size['Percentage'] = cluster_size['size'] / sum(cluster_size['size'])
cluster_size
# Plot pie chart
fig = px.pie(cluster_size, values='Percentage', names='Cluster', 
             color_discrete_sequence=px.colors.sequential.RdBu, width=800, height=500,
            title='Size of Each Cluster')
fig.show()
grouped = df.groupby(['Cluster'], as_index=False).mean().round(1)
grouped.drop(['Job'], axis=1, inplace=True)
grouped
cluster0 = df[df['Cluster']==0]
cluster1 = df[df['Cluster']==1]
cluster2 = df[df['Cluster']==2]
cluster3 = df[df['Cluster']==3]
fig, ax = plt.subplots(4,1,figsize=(10,6), constrained_layout=True, sharex=True)
ax[0].title.set_text('Cluster 0')
ax[1].title.set_text('Cluster 1')
ax[2].title.set_text('Cluster 2')
ax[3].title.set_text('Cluster 3')
ax[0].axes.xaxis.set_visible(False)
ax[1].axes.xaxis.set_visible(False)
ax[2].axes.xaxis.set_visible(False)
plt.xlabel('Age', fontsize=20)
sns.distplot(cluster0['Age'], color='darkcyan', bins=10, ax=ax[0])
sns.distplot(cluster1['Age'], color='steelblue', bins=10, ax=ax[1])
sns.distplot(cluster2['Age'], color='sandybrown', bins=10, ax=ax[2])
sns.distplot(cluster3['Age'], color='indianred', bins=10, ax=ax[3])
fig, ax = plt.subplots(4,1,figsize=(10,6), constrained_layout=True, sharex=True)
ax[0].title.set_text('Cluster 0')
ax[1].title.set_text('Cluster 1')
ax[2].title.set_text('Cluster 2')
ax[3].title.set_text('Cluster 3')
ax[0].axes.xaxis.set_visible(False)
ax[1].axes.xaxis.set_visible(False)
ax[2].axes.xaxis.set_visible(False)
plt.xlabel('Credit Amount', fontsize=20)
sns.distplot(cluster0['Credit amount'], color='darkcyan', bins=10, ax=ax[0])
sns.distplot(cluster1['Credit amount'], color='steelblue', bins=10, ax=ax[1])
sns.distplot(cluster2['Credit amount'], color='sandybrown', bins=10, ax=ax[2])
sns.distplot(cluster3['Credit amount'], color='indianred', bins=10, ax=ax[3])
fig, ax = plt.subplots(4,1,figsize=(10,6), constrained_layout=True, sharex=True)
ax[0].title.set_text('Cluster 0')
ax[1].title.set_text('Cluster 1')
ax[2].title.set_text('Cluster 2')
ax[3].title.set_text('Cluster 3')
ax[0].axes.xaxis.set_visible(False)
ax[1].axes.xaxis.set_visible(False)
ax[2].axes.xaxis.set_visible(False)
plt.xlabel('Duration', fontsize=20)
sns.distplot(cluster0['Duration'], color='darkcyan', bins=10, ax=ax[0])
sns.distplot(cluster1['Duration'], color='steelblue', bins=10, ax=ax[1])
sns.distplot(cluster2['Duration'], color='sandybrown', bins=10, ax=ax[2])
sns.distplot(cluster3['Duration'], color='indianred', bins=10, ax=ax[3])
def get_df(data):
    out = data.value_counts(normalize=True).reset_index()
    return(out)
def plot(x):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=get_df(cluster0[x])['index'],
        y=get_df(cluster0[x])[x],
        name='Cluster 0',
        marker_color='mediumaquamarine'
    ))
    fig.add_trace(go.Bar(
        x=get_df(cluster1[x])['index'],
        y=get_df(cluster1[x])[x],
        name='Cluster 1',
        marker_color='steelblue'
    ))
    fig.add_trace(go.Bar(
        x=get_df(cluster2[x])['index'],
        y=get_df(cluster2[x])[x],
        name='Cluster 2',
        marker_color='sandybrown'
    ))
    fig.add_trace(go.Bar(
        x=get_df(cluster3[x])['index'],
        y=get_df(cluster3[x])[x],
        name='Cluster 3',
        marker_color='indianred'
    ))

    fig.update_layout(barmode='group', xaxis_tickangle=45, title=x)
    fig.show()

plot('Job')
plot('Housing')
plot('Saving accounts')
plot('Checking account')
plot('Purpose')