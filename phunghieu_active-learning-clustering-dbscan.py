import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import seaborn as sns 

import plotly as py

import plotly.graph_objs as go

py.offline.init_notebook_mode(connected = True)



from sklearn.cluster import DBSCAN # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

from sklearn.preprocessing import StandardScaler # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
df = pd.read_csv(r'/kaggle/input/Mall_Customers.csv')
df.head()
df.shape
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html

df.describe()
df.dtypes
df.isnull().sum()
plt.figure(1, figsize=(15, 6))

for idx, x in enumerate(['Age', 'Annual Income (k$)', 'Spending Score (1-100)']):

    plt.subplot(1, 3, idx+1)

    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    sns.distplot(df[x], bins=20)

    plt.title('Distplot of {}'.format(x))

plt.show()
plt.figure(1, figsize = (15, 5))

sns.countplot(y = 'Gender', data = df)

plt.show()
plt.figure(1 , figsize = (15 , 7))

n = 0 

for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

    for y in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

        n += 1

        plt.subplot(3 , 3 , n)

        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

        sns.regplot(x = x , y = y , data = df)

        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )

plt.show()
plt.figure(1 , figsize = (15 , 6))

for gender in ['Male' , 'Female']:

    plt.scatter(x = 'Age' , y = 'Annual Income (k$)' , data = df[df['Gender'] == gender] ,

                s = 200 , alpha = 0.5 , label = gender)

plt.xlabel('Age'), plt.ylabel('Annual Income (k$)') 

plt.title('Age vs Annual Income w.r.t Gender')

plt.legend()

plt.show()
plt.figure(1 , figsize = (15 , 6))

for gender in ['Male' , 'Female']:

    plt.scatter(x = 'Annual Income (k$)',y = 'Spending Score (1-100)' ,

                data = df[df['Gender'] == gender] ,s = 200 , alpha = 0.5 , label = gender)

plt.xlabel('Annual Income (k$)'), plt.ylabel('Spending Score (1-100)') 

plt.title('Annual Income vs Spending Score w.r.t Gender')

plt.legend()

plt.show()
plt.figure(1 , figsize = (15 , 7))

n = 0 

for cols in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

    n += 1 

    plt.subplot(1 , 3 , n)

    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

    sns.violinplot(x = cols , y = 'Gender' , data = df , palette = 'vlag')

    sns.swarmplot(x = cols , y = 'Gender' , data = df)

    plt.ylabel('Gender' if n == 1 else '')

    plt.title('Boxplots & Swarmplots' if n == 2 else '')

plt.show()
'''Age and spending Score'''

X1 = df[['Age', 'Spending Score (1-100)']].iloc[: , :].values

X1 = StandardScaler().fit_transform(X1)

db = DBSCAN(eps=0.3, min_samples=10)



db.fit(X1)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)

core_samples_mask[db.core_sample_indices_] = True

labels = db.labels_



# Number of clusters in labels, ignoring noise if present.

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

n_noise_ = list(labels).count(-1)



print('Estimated number of clusters: %d' % n_clusters_)

print('Estimated number of noise points: %d' % n_noise_)



# #############################################################################

# Plot result

import matplotlib.pyplot as plt



# Black removed and is used for noise instead.

unique_labels = set(labels)

colors = [plt.cm.Spectral(each)

          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):

    if k == -1:

        # Black used for noise.

        col = [0, 0, 0, 1]



    class_member_mask = (labels == k)



    xy = X1[class_member_mask & core_samples_mask]

    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),

             markeredgecolor='k', markersize=14)



    xy = X1[class_member_mask & ~core_samples_mask]

    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),

             markeredgecolor='k', markersize=6)



plt.title('Estimated number of clusters: %d' % n_clusters_)

plt.show()
'''Annual Income and spending Score'''

X2 = df[['Annual Income (k$)' , 'Spending Score (1-100)']].iloc[: , :].values

X2 = StandardScaler().fit_transform(X2)

db = DBSCAN(eps=0.3, min_samples=10)



db.fit(X2)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)

core_samples_mask[db.core_sample_indices_] = True

labels = db.labels_



# Number of clusters in labels, ignoring noise if present.

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

n_noise_ = list(labels).count(-1)



print('Estimated number of clusters: %d' % n_clusters_)

print('Estimated number of noise points: %d' % n_noise_)



# #############################################################################

# Plot result

import matplotlib.pyplot as plt



# Black removed and is used for noise instead.

unique_labels = set(labels)

colors = [plt.cm.Spectral(each)

          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):

    if k == -1:

        # Black used for noise.

        col = [0, 0, 0, 1]



    class_member_mask = (labels == k)



    xy = X1[class_member_mask & core_samples_mask]

    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),

             markeredgecolor='k', markersize=14)



    xy = X1[class_member_mask & ~core_samples_mask]

    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),

             markeredgecolor='k', markersize=6)



plt.title('Estimated number of clusters: %d' % n_clusters_)

plt.show()
X3 = df[['Age' , 'Annual Income (k$)' ,'Spending Score (1-100)']].iloc[: , :].values

X3 = StandardScaler().fit_transform(X3)

db = DBSCAN(eps=0.3, min_samples=5)



db.fit(X3)

labels = db.labels_



# Number of clusters in labels, ignoring noise if present.

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

n_noise_ = list(labels).count(-1)



print('Estimated number of clusters: %d' % n_clusters_)

print('Estimated number of noise points: %d' % n_noise_)
df['labels'] = labels

trace1 = go.Scatter3d(

    x= df['Age'],

    y= df['Spending Score (1-100)'],

    z= df['Annual Income (k$)'],

    mode='markers',

    marker=dict(

        color = df['labels'], 

        size= 20,

        line=dict(

            color= df['labels'],

            width= 12

        ),

        opacity=0.8

    )

)

data = [trace1]

layout = go.Layout(

    title= 'Clusters',

    scene = dict(

        xaxis = dict(title  = 'Age'),

        yaxis = dict(title  = 'Spending Score'),

        zaxis = dict(title  = 'Annual Income')

    )

)

fig = go.Figure(data=data, layout=layout)

py.offline.iplot(fig)