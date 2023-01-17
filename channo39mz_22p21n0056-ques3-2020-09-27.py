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
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
url = '/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv'
df = pd.read_csv(url)
df = df.set_index('host_name')

df.describe()
df
df2 = df.drop(['name','neighbourhood_group','neighbourhood','room_type','last_review'],axis=1)
df2 = df2.drop(['latitude','longitude','reviews_per_month'],axis=1)
df2.info()
#lis = df2[['id']]
#lis2 = df2[['price']]
#lis3= pd.concat(lis,lis2)
true = []
lis = []
#len(df2['id'])
for y in range(40000):
    lis = []
    for i in range(1):
        #lis.append(df2['id'][y])
        lis.append(df2['price'][y])
        lis.append(df2['host_id'][y])
        lis.append(df2['minimum_nights'][y])
        lis.append(df2['number_of_reviews'][y])
        true.append(lis)
D = np.array(true)
print(D)
print(type(D))
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(D)
plt.title('New York City Airbnb Open Data Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=5)
plt.xlabel("Hotel occupancy")
plt.show()