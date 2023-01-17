# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Set the styles to Seaborn

sns.set()

# Import the KMeans module so we can perform k-means clustering with sklearn

from sklearn.cluster import KMeans

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('../input/catecorical/Categorical.csv')
data
x = data.iloc[:,1:3]
x
#clustering data

kmeans = KMeans(4)

kmeans.fit(x)
#predict cluster

identified_clusters = kmeans.fit_predict(x)

identified_clusters
data_with_clusters = data.copy()

data_with_clusters['Cluster'] = identified_clusters

data_with_clusters
plt.scatter(data['Longitude'], data['Latitude'], c=data_with_clusters['Cluster'], cmap = 'rainbow')

plt.xlim(-180,180)

plt.ylim(-90, 90)

plt.show()
data_mapped = data.copy()

data_mapped['continent'] = data_mapped['continent'].map({'North America':0,'Europe':1,'Asia':2,'Africa':3,'South America':4, 'Oceania':5,'Seven seas (open ocean)':6, 'Antarctica':7})

data_mapped
x = data_mapped.iloc[:,3:4]

x
kmeans = KMeans(7)

kmeans.fit(x)
identified_clusters = kmeans.fit_predict(x)

identified_clusters
data_with_clusters = data_mapped.copy()

data_with_clusters['Cluster'] = identified_clusters

data_with_clusters
plt.scatter(data['Longitude'], data['Latitude'], c=data_with_clusters['Cluster'], cmap = 'rainbow')

plt.xlim(-180,180)

plt.ylim(-90, 90)

plt.show()