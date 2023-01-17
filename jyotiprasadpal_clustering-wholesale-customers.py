import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



import pandas_profiling



from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
data = pd.read_csv('../input/wholesale-customers-data-set/Wholesale customers data.csv')

data.head()
data.profile_report()
scaler = StandardScaler()

scaled_df = scaler.fit_transform(data)



pd.DataFrame(scaled_df).describe()
model = KMeans(n_clusters=3,

               init='k-means++',

               n_init=10,

               max_iter=300,

               tol=0.0001,

               precompute_distances='auto',

               verbose=0,

               random_state=42,

               copy_x=True,

               n_jobs=None,

               algorithm='auto')



model.fit(scaled_df)

model.inertia_
clusters = range(1, 20)

sse=[]

for cluster in clusters:

    model = KMeans(n_clusters=cluster,

               init='k-means++',

               n_init=10,

               max_iter=300,

               tol=0.0001,

               precompute_distances='auto',

               verbose=0,

               random_state=42,

               copy_x=True,

               n_jobs=None,

               algorithm='auto')



    model.fit(scaled_df)

    sse.append(model.inertia_)



sse_df = pd.DataFrame(np.column_stack((clusters, sse)), columns=['cluster', 'SSE'])

fig, ax = plt.subplots(figsize=(13, 5))

ax.plot(sse_df['cluster'], sse_df['SSE'], marker='o')

ax.set_xlabel('Number of clusters')

ax.set_ylabel('Inertia or SSE')
model = KMeans(n_clusters=5,

               init='k-means++',

               n_init=10,

               max_iter=300,

               tol=0.0001,

               precompute_distances='auto',

               verbose=0,

               random_state=42,

               copy_x=True,

               n_jobs=-1,

               algorithm='auto')



model.fit(scaled_df)
print('SSE: ', model.inertia_)

print('\nCentroids: \n', model.cluster_centers_)



pred = model.predict(scaled_df)

data['cluster'] = pred

print('\nCount in each cluster: \n', data['cluster'].value_counts())