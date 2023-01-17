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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram
traindata = '../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv'

train = pd.read_csv(traindata)

train.head()
train.info()
x = train.iloc[:len(train)//2,9:11].values

model = AgglomerativeClustering(distance_threshold=500, n_clusters=None)

model.fit(x)
model.labels_
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
plt.title('Hierarchical Clustering Dendrogram')

# plot the top three levels of the dendrogram

plot_dendrogram(model, truncate_mode='level', p=5)

plt.xlabel("Number of points in node (or index of point if no parenthesis).")

plt.show()
fig, ax = plt.subplots(figsize=(20,7))

mydict = plot_dendrogram(model, ax=ax, truncate_mode='level', p=6, 

                leaf_font_size=10, leaf_rotation=90, show_leaf_counts=True)
train.plot(x="longitude", y="latitude", style=".", figsize=(10, 10))

plt.title("Map")

plt.ylabel("latitude")

img = plt.imread("/kaggle/input/new-york-city-airbnb-open-data/New_York_City_.png", 0)

plt.imshow(img, extent=[-74.25, -73.685, 40.49, 40.925])

plt.show()