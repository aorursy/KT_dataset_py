# Load libraries

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import random

random.seed(420)

# Load the data

data = pd.read_csv("../input/consumer-classification/cunsumer classification.csv")

data.head()
plt.scatter(data['Satisfaction'], data['Loyalty'])

plt.xlabel('Satisfaction')

plt.ylabel('Loyalty')

plt.show()
# Clustering

# Please note that I already performed k-means over the original dataset, but the performance was poor due to the non-scaling attribute. In other words, SATISFACTION ranges from 1 to 15, so the algorithm considered this feature more crucial compared to LOYALTY, which ranges from -2.6 to +2.6 

from sklearn import preprocessing

x_scaled = preprocessing.scale(data.copy())

x_scaled

random.seed(420)

kmeans_new = KMeans(4) # The value of optimal k=4 is determined based on Elbow method.

kmeans_new.fit(x_scaled) # The K-means algorithm is fitted on the scales dataset, not the original one.

clusters_new =  data.copy()

# Lets predict

clusters_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled)

clusters_new.head()

# kmeans_new.cluster_centers_
scatter = plt.scatter(clusters_new['Satisfaction'], clusters_new['Loyalty'], c=clusters_new['cluster_pred'], cmap='rainbow')

plt.title('Clustering consumers into four categories: Unfriends, Roamers, Fans, & Promoters')

plt.xlabel('Satisfaction')

plt.ylabel('Loyalty')

# classes = ['Unfriendly', 'Fans', 'Roamers', 'Promoter']; plt.legend(handles=scatter.legend_elements()[0], labels=classes, loc='upper left')

# Since the centroid of the initial cluster is determined randomly, the legends are lining randomly. I need to think about it to solve the issue. Lets skip the legend issue for now.

plt.show()
# Similar steps can be performed for SVM

# Lets practice that at home

# Further resource: https://scikit-learn.org/stable/modules/svm.html