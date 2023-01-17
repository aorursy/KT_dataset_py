# Importing Libraries

import numpy as np

import pandas as pd

from kmodes.kprototypes import KPrototypes

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
# Reading Dataset

blood = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data', sep=",",engine = 'python')
#Sanity Check

blood.head()
# standardizing data

columns_to_normalize     = ['Recency (months)','Frequency (times)','Monetary (c.c. blood)','Time (months)']

blood[columns_to_normalize] = blood[columns_to_normalize].apply(lambda x: (x - x.mean()) / np.std(x))
# Re-check after standardizing data

blood.head()
blood_matrix= blood.values

blood_matrix
# Running K-Prototype clustering

kproto = KPrototypes(n_clusters=5, init='Cao')

clusters = kproto.fit_predict(blood_matrix, categorical=[4])
kproto.cluster_centroids_
# Checking the cost of the clusters created.

kproto.cost_
# Adding the predicted clusters to the main dataset

blood['cluster_id'] = clusters
# Re-check

blood.head()
# Checking the clusters created

blooddf = pd.DataFrame(blood['cluster_id'].value_counts())

blooddf
plt.figure(figsize=(10,12))

ax=sns.barplot(x=blooddf.index, y=blooddf['cluster_id'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.show()



#Choosing optimal K

cost = []

cost = []

for cluster in list(range(2,10)):

    kproto = KPrototypes(n_clusters=cluster, init='Cao')

    kproto.fit_predict(blood_matrix, categorical=[4])

    cost.append(kproto.cost_)

plt.plot(cost);