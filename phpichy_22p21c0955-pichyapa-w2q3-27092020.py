# Hierarchy Clustering to determine recommended NY airbnb from price and availability



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# Importing Data

dt = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')



# Data cleaning

dt.drop(["id","name","host_id","host_name","last_review","reviews_per_month"],axis=1,inplace = True)

dt['neighbourhood_group'] = dt.groupby(pd.Grouper(key='neighbourhood_group')).ngroup()

dt['neighbourhood'] = dt.groupby(pd.Grouper(key='neighbourhood')).ngroup()

dt['room_type'] = dt.groupby(pd.Grouper(key='room_type')).ngroup()

dt.head()
from sklearn.preprocessing import normalize



# Normalizing Data

nm = normalize(dt)

nm = pd.DataFrame(nm, columns=dt.columns)

nm = nm.head(n=1000)



# High Value -> Less Price -> More recommended

nm["price"] = 1 - nm["price"]



# Choosing Columns -> Price & Availability

nm = nm[["price","availability_365"]]



nm
# Drawing Dendrogram to determine the number of clusters



import scipy.cluster.hierarchy as sch



dendrogram = sch.dendrogram(sch.linkage(nm, method  = "ward"))

plt.title('Dendrogram')

plt.xlabel('Price')

plt.ylabel('Availability')

plt.show()
from sklearn.cluster import AgglomerativeClustering



hc = AgglomerativeClustering(n_clusters = 3, affinity = "euclidean", linkage = "ward")

cluster = hc.fit_predict(nm)

nm["label"] = cluster



plt.figure(figsize = (15, 10))

plt.scatter(nm["price"][nm.label == 0], nm["availability_365"][nm.label == 0], color = "green")

plt.scatter(nm["price"][nm.label == 1], nm["availability_365"][nm.label == 1], color = "blue")

plt.scatter(nm["price"][nm.label == 2], nm["availability_365"][nm.label == 2], color = "red")

plt.xlabel("price")

plt.ylabel("availability")

plt.show()



# Blue -> Recommended | Green -> Average | Red -> Not Recommended