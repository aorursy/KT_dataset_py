# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib.pyplot import plot

from sklearn.preprocessing import MinMaxScaler

import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
block= 'block_58'

original_data = pd.read_csv('/kaggle/input/daily_dataset/daily_dataset/{}.csv'.format(block))

original_data.describe()
max_data = original_data.loc[original_data.groupby('LCLid').pipe(lambda group: group.energy_max.idxmax(skipna=True))][['LCLid','day','energy_max']]

min_data = original_data.loc[original_data.groupby('LCLid').pipe(lambda group: group.energy_max.idxmin(skipna=True))][['LCLid','day','energy_min']]

# original_data.describe()



data = original_data.groupby('LCLid').agg({'energy_median': ['mean'], 'energy_mean': ['mean'], 'energy_sum': ['sum']})

data = data.merge(max_data, left_on='LCLid', right_on='LCLid',suffixes=('_left', '_max'))

data = data.merge(min_data, left_on='LCLid', right_on='LCLid',suffixes=('_max', '_min'))

data['day_max'] = pd.to_datetime(data['day_max']).dt.dayofweek

data['day_min'] = pd.to_datetime(data['day_min']).dt.dayofweek



# data = pd.concat([data,pd.get_dummies(data['day_min'], prefix='day_min')],axis=1).drop(columns=['day_min'])

# data = pd.concat([data,pd.get_dummies(data['day_max'], prefix='day_max')],axis=1).drop(columns=['day_max'])

data['min_max_ratio'] = data.pipe(lambda group: group.energy_min/ group.energy_max)

# print(original_data[(original_data['LCLid']=='MAC000094')])





data.head()
from sklearn.cluster import KMeans

from sklearn.metrics import pairwise_distances_argmin_min

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



scaler = MinMaxScaler(feature_range=(0, 1))

X = scaler.fit_transform(data.drop(columns=['LCLid']))



Nc = range(1, 20)

kmeans = [KMeans(n_clusters=i) for i in Nc]

score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]



plt.plot(Nc,score)

plt.xlabel('Number of Clusters')

plt.ylabel('Score')

plt.title('Elbow Curve')

plt.show()
kmeans = KMeans(n_clusters=4).fit(X)

centroids = kmeans.cluster_centers_

print(centroids)
clustered_data = pd.concat([data,pd.DataFrame(kmeans.predict(X))],axis=1).sort_values(by=[0])

clustered_data = clustered_data.rename(columns={0: 'Cluster'})

clustered_data.head()
hourly_data = pd.read_csv('/kaggle/input/hhblock_dataset/hhblock_dataset/{}.csv'.format(block))



def create_plot(LCLid, start_date=None, end_date=None, ax=None, title=None):

    plot_data = hourly_data[(hourly_data['LCLid']==LCLid)].set_index('day').drop(columns=['LCLid'])

    plot_data = plot_data.stack().reset_index().rename(columns={0: LCLid,'level_1': 'time','day':'date'})

    plot_data['time'] = plot_data['time'].apply(lambda x: x.replace('hh_',''))

    plot_data['datetime'] = plot_data[['date','time']].apply(lambda x: pd.to_datetime(x['date'])+datetime.timedelta(minutes=int(x['time'])*30),axis=1)

    plot_data = plot_data.drop(columns=['date','time'])

    

    if start_date:

        plot_data = plot_data[(plot_data['datetime'] >= pd.to_datetime(start_date))] 

    if end_date:

        plot_data = plot_data[(plot_data['datetime'] <= pd.to_datetime(end_date))]  

        

    if len(plot_data.index>0):

        plot_data = plot_data.set_index('datetime')

        plot = plot_data.plot(ax=ax, title=title, figsize = (20,6))

        print("Processing {LCLid}, {title}".format(**{'LCLid':LCLid, 'title':title}))

        return plot

    

    print("[WARNING] LCLid: {LCLid} has no data between dates: {start_date}-> {end_date}".format(**{'LCLid': LCLid, 'start_date': start_date, 'end_date': end_date}))

    return None

    



# hourly_data.head()

ax = None

cluster = 0

start_date='2014-02-01'

end_date='2014-02-07'

title = "Cluster: {}".format(str(cluster))

for index, consumer_row in clustered_data[['LCLid','Cluster']].iterrows():

    plot_new_cluster = cluster != consumer_row['Cluster']

    consumer_id, cluster = consumer_row['LCLid'], consumer_row['Cluster']

    aux = create_plot(consumer_id,start_date=start_date,end_date=end_date,ax=ax,title=title)

    if aux:

        ax = aux

    if plot_new_cluster:

        ax = None

        title = "Cluster: {}".format(str(cluster))
