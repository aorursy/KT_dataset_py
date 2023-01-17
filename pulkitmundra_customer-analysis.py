import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #data visualisation

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/Online Retail.csv")
df.head()
df.shape
df.describe()
df = df.loc[df["Quantity"] > 0]

df.shape
df.info()
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df.info()
df["Sale"] = df.Quantity * df.UnitPrice
df.head()
monetry = df.groupby("CustomerID").Sale.sum()

monetry = monetry.reset_index()

monetry.head()
frequency = df.groupby("CustomerID").InvoiceNo.count()

frequency = frequency.reset_index()

frequency.head()
LastDate = max(df.InvoiceDate)

LastDate = LastDate + pd.DateOffset(days = 1)

df["Diff"] = LastDate - df.InvoiceDate
recency = df.groupby("CustomerID").Diff.min()

recency = recency.reset_index()

recency.head()
rmf = monetry.merge(frequency, on = "CustomerID")

rmf = rmf.merge(recency, on = "CustomerID")

rmf.columns = ["CustomerID", "Monetry", "Frequence", "Recency"]

rmf
RMF_km = rmf.drop("CustomerID", axis = 1)

RMF_km.Recency = RMF_km.Recency.dt.days

RMF_km.head()
RMF_km = pd.DataFrame(RMF_km, columns=["Monetry", "Frequence", "Recency"])
from sklearn.cluster import KMeans
ssd = []

for k in range(1, 20):

    km = KMeans(n_clusters=k)

    km.fit(RMF_km)

    ssd.append(km.inertia_)
plt.plot(np.arange(1,20), ssd, color = "green")

plt.scatter(np.arange(1,20), ssd, color = "brown")

plt.show()
model = KMeans(n_clusters=5)

ClusterID = model.fit_predict(RMF_km)
RMF_km["ClusterID"] = ClusterID

RMF_km
km_cluster_sale = RMF_km.groupby("ClusterID").Monetry.mean()

km_cluster_Recency = RMF_km.groupby("ClusterID").Recency.mean()

km_cluster_Frequence = RMF_km.groupby("ClusterID").Frequence.mean()
import seaborn as sns
fig, axs = plt.subplots(1,3, figsize = (15, 5))

sns.barplot(x = [0,1,2,3,4],  y = km_cluster_sale , ax = axs[0])

sns.barplot(x = [0,1,2,3,4],  y = km_cluster_Frequence , ax = axs[1])

sns.barplot(x = [0,1,2,3,4],  y = km_cluster_Recency , ax = axs[2])
fig, axs = plt.subplots(1,3, figsize = (15, 5))



ax1 = fig.add_subplot(1, 3, 1)

plt.title("Sale Mean")

ax1.pie(km_cluster_sale, labels = [0,1,2,3,4])



ax1 = fig.add_subplot(1, 3, 2)

plt.title("Frequency Mean")

ax1.pie(km_cluster_Frequence, labels = [0,1,2,3,4])



ax1 = fig.add_subplot(1, 3, 3)

plt.title("Recency Mean")

ax1.pie(km_cluster_Recency, labels = [0,1,2,3,4])







plt.axis("off")

plt.show()