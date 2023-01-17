import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from sklearn.cluster import KMeans
raw_data = pd.read_csv("../input/oura-smart-ring-sleep-data/oura_sleep_data.csv")
data = raw_data.copy()

data.head()
plt.scatter(data['deep_sleep'], data['rem_sleep'])

plt.xlim(0, 17000)

plt.ylim(0, 12000)

plt.show()
x = data.iloc[:,4:6] #Counts from FIRST column to SECOND

#and prints those

x
kmeans = KMeans(5)
kmeans.fit(x)
identified_clusters = kmeans.fit_predict(x)

identified_clusters
data_with_clusters = data.copy()

data_with_clusters['Cluster'] = identified_clusters

data_with_clusters
plt.scatter(data['deep_sleep'], data['rem_sleep'],c=data_with_clusters['Cluster'], cmap = 'rainbow')

plt.xlim(0, 15000)

plt.ylim(0, 16000)

plt.show()
plt.scatter(data['deep_sleep'], data['rem_sleep'],c=data_with_clusters['Cluster'], cmap = 'rainbow')

plt.xlim(0, 15000)

plt.ylim(0, 20000)

plt.show()
#Use the integrated sklearn method 'inertia_'

kmeans.inertia_
#Write a loop that calculates and saves the WCSS for any number of clusters from 1 up to 10 (or more if you wish).

wcss = []

# 'cl_num' is a that keeps track the highest number of clusters we want to use the WCSS method for.

# Note that 'range' doesn't include the upper boundery

cl_num = 11

for i in range (1,cl_num):

    kmeans= KMeans(i)

    kmeans.fit(x)

    wcss_iter = kmeans.inertia_

    wcss.append(wcss_iter)

wcss
number_clusters = range(1,cl_num)

plt.plot(number_clusters, wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('Within-cluster Sum of Squares')
kmeans = KMeans(2)

kmeans.fit(x)
identified_clusters = kmeans.fit_predict(x)
data_with_clusters = data.copy()

data_with_clusters['Cluster'] = identified_clusters
plt.scatter(raw_data['deep_sleep'], raw_data['rem_sleep'], c=data_with_clusters['Cluster'], cmap = 'rainbow')

plt.xlim(0, 15000)

plt.ylim(0, 20000)

plt.show()
kmeans = KMeans(3)

kmeans.fit(x)
identified_clusters = kmeans.fit_predict(x)
data_with_clusters = data.copy()

data_with_clusters['Cluster'] = identified_clusters
plt.scatter(raw_data['deep_sleep'], raw_data['rem_sleep'], c=data_with_clusters['Cluster'], cmap = 'rainbow')

plt.xlim(-1000, 15000)

plt.ylim(-1000, 20000)

plt.show()
import pandas as pd

oura_sleep_data = pd.read_csv("../input/oura-smart-ring-sleep-data/oura_sleep_data.csv")