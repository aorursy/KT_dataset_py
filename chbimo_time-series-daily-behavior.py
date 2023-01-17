import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.decomposition import PCA

from sklearn.mixture import GaussianMixture



import plotly

from plotly.graph_objs import Scatter, Layout

plotly.offline.init_notebook_mode(connected=True)
data = pd.read_csv('../input/train.csv')

data = data[['pickup_datetime']]
data['pickup_datetime'] = data['pickup_datetime'].map(lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))
data['count'] = data['pickup_datetime'].map(lambda x:1)
data = data.set_index('pickup_datetime', drop=True)
hourly = data.resample('h').sum()

hourly.head()
fig, ax = plt.subplots()

hourly[['count']].plot(ax=ax)

ax.set_xlabel('date')

ax.set_ylabel('counts')
pivoted = hourly[['count']].pivot_table(index=hourly.index.date,

                                     columns=hourly.index.hour)

pivoted = pivoted.fillna(0)

pivoted.head()
Xpca = PCA(n_components=2).fit_transform(pivoted)

Xpca.shape
plt.scatter(Xpca[:,0], Xpca[:,1])
n_cluster=2



gmm = GaussianMixture(n_cluster, covariance_type='full', random_state=0)

gmm.fit(Xpca)

cluster_label = gmm.predict(Xpca)

plt.scatter(Xpca[:, 0], Xpca[:, 1], c=cluster_label)
pivoted['cluster'] = cluster_label

hourly = hourly.join(pivoted['cluster'],on=hourly.index.date)

hourly.head()
by_hour = hourly.groupby(['cluster', hourly.index.time]).mean()

by_hour.head()
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

hourly_ticks = 4 * 60 * 60 * np.arange(6)



for i in range(2):

    by_hour['count'].loc[i].plot(ax=ax[i], xticks=hourly_ticks)

    ax[i].set_title('Cluster {0}'.format(i))

    ax[i].set_ylabel('average hourly trips')
dayofweek = pd.to_datetime(pivoted.index).dayofweek

plt.scatter(Xpca[:, 0], Xpca[:, 1], c=dayofweek,

            cmap=plt.cm.get_cmap('jet', 7))

cb = plt.colorbar(ticks=range(7))

cb.set_ticklabels(['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])

plt.clim(-0.5, 6.5);


pivoted['xpca0'] = Xpca[:,0]

pivoted['xpca1'] = Xpca[:,1]

pivoted['day'] = dayofweek



day = {0:'red',

       1:'green',

      2:'blue',

      3:'cyan',

      4:'magenta',

      5:'yellow',

      6:'black'}



pivoted['day'] = pivoted['day'].map(lambda x: day[x])



plotly.offline.iplot({

        'data':[Scatter(x=pivoted['xpca0'], y=pivoted['xpca1'],text=pivoted.index,mode='markers',

                       marker=dict(color=pivoted['day']))],

    })