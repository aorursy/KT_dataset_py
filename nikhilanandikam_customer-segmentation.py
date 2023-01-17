import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df=pd.read_csv("../input/Online Retail.csv")
#for finding the shape of data

df.shape  
#To find column names

df.keys()
#data description

df.describe()
df=df[df["Quantity"]>0]

df.shape

df.describe()
df.head()
#converting InvoiceDate to datetime format

df["InvoiceDate"]=pd.to_datetime(df.InvoiceDate)

df.info()
df["Sale"]=df.Quantity*df.UnitPrice

df.head()
#Data is grouped by CustomerID and total amount of sale gives monetary

monetry = df.groupby("CustomerID").Sale.sum()

monetry = monetry.reset_index()

monetry.head()
#data is grouped by CustomerID and count of the InvoiceNo gives frequency

frequency = df.groupby("CustomerID").InvoiceNo.count()

frequency = frequency.reset_index()

frequency.head()
#LastDate in the inventory is calculated

LastDate = max(df.InvoiceDate)

LastDate = LastDate + pd.DateOffset(days = 1)
#Difference between the Last date and the InvoiceDate of each row 

df["Diff"] = LastDate - df.InvoiceDate
#data is grouped byCustomerID and then Minimum diff gives recency

recency = df.groupby("CustomerID").Diff.min()

recency = recency.reset_index()

recency.head()
#Recency,Frequency and Monetary data frames are merged to form Single dataFrame

rmf = monetry.merge(frequency, on = "CustomerID")

rmf = rmf.merge(recency, on = "CustomerID")

rmf.columns = ["CustomerID", "Monetry", "Frequence", "Recency"]

rmf
RMF1 = rmf.drop("CustomerID", axis = 1)

RMF1.Recency = RMF1.Recency.dt.days

RMF1.head()
#from sklearn.preprocessing import StandardScaler

#ss = StandardScaler()

#RMF1 = ss.fit_transform(RMF1)

#RMF1
# Convert Data IN DataFrame

RMF1 = pd.DataFrame(RMF1, columns=["Monetry", "Frequence", "Recency"])

RMF1
from sklearn.cluster import KMeans
#Training the data and fitting 

#Finding inertias for a range of k values

ssd = []

for k in range(1, 20):

    km = KMeans(n_clusters=k)

    km.fit(RMF1)

    ssd.append(km.inertia_)
#Plotting the inertia's wrt 'k' values 

plt.plot(np.arange(1,20), ssd, color = "green")

plt.scatter(np.arange(1,20), ssd, color = "brown")

plt.show()
#Training with KMeans and predicting the ClusterID.

model = KMeans(n_clusters=5)

ClusterID = model.fit_predict(RMF1)
RMF1["ClusterID"] = ClusterID

RMF1
km_cluster_sale = RMF1.groupby("ClusterID").Monetry.mean()

km_cluster_Recency = RMF1.groupby("ClusterID").Recency.mean()

km_cluster_Frequence = RMF1.groupby("ClusterID").Frequence.mean()
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