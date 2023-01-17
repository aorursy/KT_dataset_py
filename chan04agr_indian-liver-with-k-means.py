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
df_original = pd.read_csv("/kaggle/input/indian-liver-patient-records/ILPD.csv")
df.head()
df

df.columns
df_label = df_original.is_patient
df = df_original.drop('is_patient', axis=1)

df.head()

lis = []

for gen in df.gender:

    if gen == "Male":

        lis.append(1)

    else:

        lis.append(0)



df['sex'] = lis
df = df.drop('gender', axis=1)



df.head()
df = (df-df.mean())/df.std()
df.head()
df = (df-df.mean())/df.std()

df.head()
df.count()
from sklearn.utils import shuffle

df = shuffle(df)
df.head(10)
train_data = df.as_matrix()[0:450, :]

test_data = df.as_matrix()[450:583, :]
test_data.shape

# Using scikit-learn to perform K-Means clustering

from sklearn.cluster import KMeans

    

# Specify the number of clusters (3) and fit the data X

kmeans = KMeans(n_clusters=2, random_state=0).fit(train_data)
# Get the cluster centroids

print(kmeans.cluster_centers_)

    

# Get the cluster labels

print(kmeans.labels_)
labels = df_label.as_matrix()[0:450]

labels
label1 = labels - 1
acc = label1 == kmeans.labels_

acc.sum()/450
label2 = np.ones(labels.shape)

label2[labels == 2] = 0

label2
acc2 = label2 == kmeans.labels_

acc2.sum()/450
from sklearn.metrics import silhouette_score



print(silhouette_score(train_data, kmeans.labels_))
(labels==2).sum()/450