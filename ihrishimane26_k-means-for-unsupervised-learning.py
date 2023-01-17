import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("../input/CC GENERAL.csv")
df.head()
df.info()
df.describe()
df.corr()
df.isna().sum()
df=df.fillna(df.median())
df.isna().sum()
from sklearn.cluster import KMeans
df.drop('CUST_ID',axis=1,inplace=True)
df.info()
sse_ = []
for k in range(1, 15):
    kmeans = KMeans(n_clusters=k).fit(df)
    sse_.append([k, kmeans.inertia_])
sse_
plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1])
from sklearn.metrics import silhouette_score
sse_ = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k).fit(df)
    sse_.append([k, silhouette_score(df, kmeans.labels_)])
plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1]);
kmeans=KMeans(n_clusters=8)
kmeans.fit(df)
kmeans.cluster_centers_
y_kmeans = kmeans.predict(df)
y_kmeans
df["cluster"] = y_kmeans
df.head()