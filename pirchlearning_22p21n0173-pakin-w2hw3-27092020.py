# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
AB_NYC_2019 = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
AB_NYC_2019.head()
AB_NYC_2019.isnull().sum()
AB_NYC_2019 = AB_NYC_2019.drop(['id', 'host_id', 'name', 'host_name', 'last_review'], axis=1)
AB_NYC_2019 = AB_NYC_2019.fillna({'reviews_per_month':0})
AB_NYC_2019.isnull().sum()
AB_NYC_2019 = AB_NYC_2019.astype({'neighbourhood_group':'category', 'neighbourhood':'category', 'room_type':'category'})
AB_NYC_2019.neighbourhood_group = AB_NYC_2019.neighbourhood_group.cat.codes
AB_NYC_2019.neighbourhood = AB_NYC_2019.neighbourhood.cat.codes
AB_NYC_2019.room_type = AB_NYC_2019.room_type.cat.codes
from sklearn.preprocessing import normalize
AB_NYC_2019_scaled = normalize(AB_NYC_2019)
AB_NYC_2019_scaled = pd.DataFrame(AB_NYC_2019_scaled, columns=AB_NYC_2019.columns)
AB_NYC_2019_scaled.head()
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    linkage_matrix
    dendrogram(linkage_matrix, **kwargs)
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model = model.fit(AB_NYC_2019_scaled.sample(n=int(AB_NYC_2019_scaled.shape[0]/2), random_state=13))
plt.title('Hierarchical Clustering Dendrogram')
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
