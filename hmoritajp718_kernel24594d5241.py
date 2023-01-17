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
data = pd.read_csv("/kaggle/input/bank-marketing/bank-additional-full.csv", delimiter=";")
data.head()
data_pos = data[data["y"]=='yes']
data_neg = data[data["y"]=='no']

from sklearn.utils import shuffle
balanced_data = shuffle(pd.concat([data_pos, data_neg.sample(len(data_pos))]))
small_balanced_data = shuffle(pd.concat([data_pos.sample(500), data_neg.sample(500)]))

# data = balanced_data
data = small_balanced_data

Y = (data["y"]=="yes")*1
data.info()
data.drop('y', axis=1, inplace = True)
data['age'].unique()
from sklearn.preprocessing import LabelEncoder
categorical_column = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                      'day_of_week', 'poutcome']
for i in categorical_column:
    le = LabelEncoder()
    data[i] = le.fit_transform(data[i])
print(data.head())
# Dropping duration of call because it creates a heavy bias as pointed in original dataset.
data.drop('duration', inplace = True, axis=1)
data.head()
### k-means ###
from sklearn.cluster import KMeans # K-means クラスタリングをおこなう
# この例では 3 つのグループに分割
kmeans_model = KMeans(n_clusters=3).fit(data)
# 分類結果のラベルを取得する
kmeans_labels = pd.DataFrame(kmeans_model.labels_)[0]
print(kmeans_labels.value_counts()/kmeans_labels.size) # % of clusterd group: kmeans
# import hierarchical clustering libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(data, method='ward'))
# create clusters
hc = AgglomerativeClustering(n_clusters=10, affinity = 'euclidean', linkage = 'ward')
# save clusters for chart
hc_model = hc.fit_predict(data)
hc_labels = pd.DataFrame(hc_model)[0]
print(hc_labels.value_counts()/hc_labels.size) # % of clusterd group: hierarchical clustering
