# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data1 = pd.read_excel('../input/lemon-trade/lemon.xlsx')
data1.info()
data1.head()
data1.describe()
data1.columns
data1_countries = pd.DataFrame(data1.iloc[:,0])

data1_features = data1.iloc[:,1:]
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

imputer.fit(data1_features)

data1_features_imputed = imputer.transform(data1_features)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

data1_features_scaled = sc.fit_transform(data1_features_imputed)

data1_df = pd.DataFrame(data1_features_scaled, columns = ["Import value ($)", "Annual increase rate in import (%)","Quarterly increase rate in import (%)", "Import unit value (USD/ton)",

"Turkey's export value ($)", "Annual increase rate in Turkey's export (%)","Quarterly increase rate in Turkish export (%)","Turkey's export unit value (USD/ton)"])
data1_df.head()
#In this round we will use import values of countries and Turkey's export value for first clustering

x1 = data1_df.loc[:,["Import value ($)","Turkey's export value ($)"]].values
print((x1))
from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):

  kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)  # init = 'k-means++' is used to avoid random initilization trap

  kmeans.fit(x1)

  wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)  # init = 'k-means++' is used to avoid random initilization trap

y1 = kmeans.fit_predict(x1)
print(y1)
plt.scatter(x1[y1 == 0, 0], x1[y1 == 0, 1], s = 100, c = 'blue', label = 'Cluster 1' )

plt.scatter(x1[y1 == 1, 0], x1[y1 == 1, 1], s = 100, c = 'red', label = 'Cluster 2' )

plt.scatter(x1[y1 == 2, 0], x1[y1 == 2, 1], s = 100, c = 'green', label = 'Cluster 3' )

plt.scatter(x1[y1 == 3, 0], x1[y1 == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4' )

plt.scatter(x1[y1 == 4, 0], x1[y1 == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5' )

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 10, c = 'orange', label = 'Centroids')

plt.title('Clusters_1')

plt.xlabel('Import Value($)')

plt.ylabel("Turkey's Export Value")

plt.legend()

plt.show()
data1_df["Clusters_1"] = y1

data1_final = pd.concat([data1_countries, data1_df], sort=False, axis=1)

print("Clusters According to Import and TR Export Values")

data1_final.head()
# In this round we will further subgroup clusters 2,3 and 5 of the first round

clusters_1 = [1,2,4]  # clusters 2,3,5

data2 = data1_final[data1_final.Clusters_1.isin(clusters_1)]    # filtering
data2.info()
data2_countries = pd.DataFrame(data2.iloc[:,0])

data2_df = data2.iloc[:,1:]
#In this round we will use annual increase rate in import values of countries and Turkey's export value for second clustering

x2 = data2_df.loc[:,["Annual increase rate in import (%)","Annual increase rate in Turkey's export (%)"]].values
print(x2)
from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):

  kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)  # init = 'k-means++' is used to avoid random initilization trap

  kmeans.fit(x2)

  wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)  # init = 'k-means++' is used to avoid random initilization trap

y2 = kmeans.fit_predict(x2)
print(y2)
plt.scatter(x2[y2 == 0, 0], x2[y2 == 0, 1], s = 100, c = 'blue', label = 'Cluster 1' )

plt.scatter(x2[y2 == 1, 0], x2[y2 == 1, 1], s = 100, c = 'red', label = 'Cluster 2' )

plt.scatter(x2[y2 == 2, 0], x2[y2 == 2, 1], s = 100, c = 'green', label = 'Cluster 3' )

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 10, c = 'orange', label = 'Centroids')

plt.title('Clusters_2')

plt.xlabel('Annual increase rate in import(%)')

plt.ylabel("Annual increase rate in Turkey's export (%)")

plt.legend()

plt.show()
data2_df["Clusters_2"] = y2

data2_final = pd.concat([data2_countries, data2_df], sort=False, axis=1)

print("Clusters According to Annual Increase Rates")

data2_final.head()
# In this round we will further subgroup clusters 2 and 3 of the second round

clusters_2 = [1,2]  # clusters 2,3

data3 = data2_final[data2_final.Clusters_2.isin(clusters_2)]    # filtering
data3.info()
data3_countries = pd.DataFrame(data3.iloc[:,0])

data3_df = data3.iloc[:,1:]
#In this round we will use import unit value of countries and export unit value for Turkey for third clustering

x3 = data3_df.loc[:,["Import unit value (USD/ton)","Turkey's export unit value (USD/ton)"]].values
print(x3)
from sklearn.cluster import KMeans

wcss = []

for i in range(1,6):

  kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)  # init = 'k-means++' is used to avoid random initilization trap

  kmeans.fit(x3)

  wcss.append(kmeans.inertia_)

plt.plot(range(1,6), wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)  # init = 'k-means++' is used to avoid random initilization trap

y3 = kmeans.fit_predict(x3)
print(y3)
plt.scatter(x3[y3 == 0, 0], x3[y3 == 0, 1], s = 100, c = 'blue', label = 'Cluster 1' )

plt.scatter(x3[y3 == 1, 0], x3[y3 == 1, 1], s = 100, c = 'red', label = 'Cluster 2' )

plt.scatter(x3[y3 == 2, 0], x3[y3 == 2, 1], s = 100, c = 'green', label = 'Cluster 3' )

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 10, c = 'orange', label = 'Centroids')

plt.title('Clusters_3')

plt.xlabel('Import unit value (USD/ton)')

plt.ylabel("Turkey's export unit value (USD/ton)")

plt.legend()

plt.show()
data3_df["Clusters_3"] = y3

data3_final = pd.concat([data3_countries, data3_df], sort=False, axis=1)

print("Clusters According to Unit Values")

data3_final.head()
# Clusters with respectively higher unit values for Turkish Export are chosen as the target markets

clusters_3 = [0,2]  # clusters 1,3

data_final = data3_final[data3_final.Clusters_3.isin(clusters_3)]    # filtering

data_final.head()
plt.scatter(x3[y3 == 0, 0], x3[y3 == 0, 1], s = 100, c = 'blue', label = 'Cluster 1' )

plt.scatter(x3[y3 == 2, 0], x3[y3 == 2, 1], s = 100, c = 'green', label = 'Cluster 3' )

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 10, c = 'orange', label = 'Centroids')

plt.title('Clusters_3')

plt.xlabel('Import unit value (USD/ton)')

plt.ylabel("Turkey's export unit value (USD/ton)")

plt.legend()

plt.show()