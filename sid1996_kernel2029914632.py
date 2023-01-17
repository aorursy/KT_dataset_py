#Importing Libraries

%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale

from sklearn.cluster import KMeans

import seaborn as sns

from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree
#reading Dataset

retail = pd.read_excel("Online Retail.xlsx")
# parse date

#We are using infer_datetime_format=True to read parse the date data, this is slower than using a pre-defined format.

retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'], infer_datetime_format=True)
#Sanity Check

retail.head()
retail.shape
retail.describe()
retail.info()
#Na Handling

retail.isnull().values.any()
#count the missing Values

retail.isnull().values.sum()
#Calculate the percentage of Missing Values

retail.isnull().sum()*100/retail.shape[0]
#dropping the na cells

order_wise = retail.dropna()
#Sanity check

order_wise.shape

order_wise.isnull().sum()
#RFM implementation

#Monetary 

#create a new variables called amount

amount  = pd.DataFrame(order_wise.Quantity * order_wise.UnitPrice, columns = ["Amount"])
amount.head()
#merging amount in order_wise

order_wise = pd.concat(objs = [order_wise, amount], axis = 1, ignore_index = False)

order_wise.head()
#Monetary Function

monetary = order_wise.groupby("CustomerID").Amount.sum()

monetary = monetary.reset_index()
monetary.head()
#Frequency function

frequency = order_wise[['CustomerID', 'InvoiceNo']]
frequency.head()
k = frequency.groupby("CustomerID").InvoiceNo.count()

k = pd.DataFrame(k)

k = k.reset_index()

k.columns = ["CustomerID", "Frequency"]

k.head()
#creating master dataset

master = monetary.merge(k, on = "CustomerID", how = "inner")

master.head()
#Generating recency function

recency  = order_wise[['CustomerID','InvoiceDate']]

recency.head()
maximum = max(recency.InvoiceDate)

maximum = maximum + pd.DateOffset(days=1)

recency['diff'] = maximum - recency.InvoiceDate

recency.head()
#Dataframe merging by recency

df = pd.DataFrame(recency.groupby('CustomerID').diff.min())

df = df.reset_index()

df.columns = ["CustomerID", "Recency"]

df.head()
#Combining all recency, frequency and monetary parameters

RFM = k.merge(monetary, on = "CustomerID")

RFM = RFM.merge(df, on = "CustomerID")

RFM.head()
# outlier treatment for Amount

plt.boxplot(RFM.Amount)

Q1 = RFM.Amount.quantile(0.25)

Q3 = RFM.Amount.quantile(0.75)

IQR = Q3 - Q1

RFM = RFM[(RFM.Amount >= Q1 - 1.5*IQR) & (RFM.Amount <= Q3 + 1.5*IQR)]
# outlier treatment for Frequency

plt.boxplot(RFM.Frequency)

Q1 = RFM.Frequency.quantile(0.25)

Q3 = RFM.Frequency.quantile(0.75)

IQR = Q3 - Q1

RFM = RFM[(RFM.Frequency >= Q1 - 1.5*IQR) & (RFM.Frequency <= Q3 + 1.5*IQR)]
# outlier treatment for Recency

plt.boxplot(RFM.Recency)

Q1 = RFM.Recency.quantile(0.25)

Q3 = RFM.Recency.quantile(0.75)

IQR = Q3 - Q1

RFM = RFM[(RFM.Recency >= Q1 - 1.5*IQR) & (RFM.Recency <= Q3 + 1.5*IQR)]
RFM.head(20)
# standardise all parameters

RFM_norm1 = RFM.drop("CustomerID", axis=1)

RFM_norm1.Recency = RFM_norm1.Recency.dt.days
RFM_norm1.Recency.head()
from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()

standard_scaler.fit_transform(RFM_norm1)
# sum of squared distances

ssd = []

for num_clusters in list(range(1,21)):

    model_clus = KMeans(n_clusters = num_clusters, max_iter=50)

    model_clus.fit(RFM_norm1)

    ssd.append(model_clus.inertia_)



plt.plot(ssd)
model_clus5 = KMeans(n_clusters = 5, max_iter=50)

model_clus5.fit(RFM_norm1)
# analysis of clusters formed

RFM.index = pd.RangeIndex(len(RFM.index))

RFM_km = pd.concat([RFM, pd.Series(model_clus5.labels_)], axis=1)

RFM_km.columns = ['CustomerID', 'Frequency', 'Amount', 'Recency', 'ClusterID']



RFM_km.Recency = RFM_km.Recency.dt.days

km_clusters_amount = 	pd.DataFrame(RFM_km.groupby(["ClusterID"]).Amount.mean())

km_clusters_frequency = 	pd.DataFrame(RFM_km.groupby(["ClusterID"]).Frequency.mean())

km_clusters_recency = 	pd.DataFrame(RFM_km.groupby(["ClusterID"]).Recency.mean())
df = pd.concat([pd.Series([0,1,2,3,4]), km_clusters_amount, km_clusters_frequency, km_clusters_recency], axis=1)

df.columns = ["ClusterID", "Amount_mean", "Frequency_mean", "Recency_mean"]

df.head()
sns.barplot(x=df.ClusterID, y=df.Amount_mean)

sns.barplot(x=df.ClusterID, y=df.Frequency_mean)
sns.barplot(x=df.ClusterID, y=df.Recency_mean)