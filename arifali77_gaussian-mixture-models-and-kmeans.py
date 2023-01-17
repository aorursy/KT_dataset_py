# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Clustering_gmm.csv')

data.head()
import matplotlib.pyplot as plt

%matplotlib inline
plt.figure(figsize=(7,7))

plt.scatter(data['Weight'], data['Height'])

plt.xlabel('Weight')

plt.ylabel('Height')

plt.title('data distribution')

plt.show()
# training kmeans model

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4)

kmeans.fit(data)



# predictions from kmeans

pred = kmeans.predict(data)

frame = pd.DataFrame(data)

frame['cluster'] = pred

frame.columns = ['Weight', 'Height', 'cluster']



# plotting results

color = ['red', 'blue', 'cyan', 'green']

for k in range(0,4):

    data = frame [frame['cluster'] == k]

    plt.scatter(data['Weight'], data['Height'], c=color[k])

plt.show()
data = pd.read_csv('../input/Clustering_gmm.csv')

data.head()
# training gaussian model 



from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=4)

gmm.fit(data)



# prediction of gmm



labels = gmm.predict(data)

frame = pd.DataFrame(data)

frame['cluster'] = labels

frame.columns = ['Weight','Height', 'cluster']



color = ['orange', 'blue', 'pink', 'green']



for k in range (0,4):

    data = frame[frame['cluster'] == k]

    plt.scatter(data['Weight'], data['Height'], c=color[k])

plt.show()