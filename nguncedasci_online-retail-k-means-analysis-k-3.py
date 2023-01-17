import pandas as pd
import numpy as np
import seaborn as sns
pd.set_option("display.max_columns",None);
pd.set_option("display.max_rows",None);
retail=pd.read_excel("../input/online-retail-ii-dataset/online_retail_II.xlsx", sheet_name="Year 2010-2011")

df=retail.copy()
df.head(5)
df.info()
df.isna().sum()  
import missingno as msno
msno.bar(df);
df[df.isnull().any(axis=1)].shape
import missingno as msno
msno.heatmap(df);
df[df.isnull().any(axis=1)].shape
135080/541910
missingdf=df[(df["Description"].isnull()==True)  & (df["Customer ID"].isnull()==True)].head()
missingdf.head()
missingdf.groupby("Country").agg({"Price":np.mean})
df=df[df.notnull().all(axis=1)]
print(df.shape)
df.head()
df["Customer ID"]=df["Customer ID"].astype("int64")
df.head(1)
df[df["Invoice"].astype("str").str.get(0)=="C"].shape
df[df["Invoice"].astype("str").str.get(0)=="C"].head()
df=df[df["Invoice"].astype("str").str.get(0)!="C"]
df[df["Invoice"].astype(str).str.get(0)!="5"].head()
df[df["Quantity"]<0]
df.info()
df.Country.value_counts().head()
df[df.duplicated(["Description","Invoice"],keep=False)].head()
df=df.drop([125])
df[df.duplicated(["Description","Invoice"],keep=False)].head()
df.groupby("Country").agg({"Price":"sum"}).applymap('{:,.2f}'.format).sort_values(by="Price", ascending=True).head(10)
# Unique products
df["Description"].nunique()
# Each products counts are..
df.Description.value_counts().head() # counts of categorical values
# Best-seller
df.groupby("Description").agg({"Quantity":sum}).sort_values(by="Quantity", ascending=False).head()
# Unique invoice
df["Invoice"].nunique()
df["Total_price"]=df["Quantity"]*df["Price"]
df.head(5)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
# The top invoices for price
df.groupby("Invoice").agg({"Total_price":sum}).head()
df.groupby("Invoice").agg({"Total_price":"sum"}).sort_values("Total_price", ascending=False).head(11)
# The highest invoice
df[df["Invoice"]==581483]
# The most expensive product is "POSTAGE"
df.sort_values("Price",ascending=False).head()
# Countries total prices
df.groupby("Country").agg({"Total_price":"sum"}).sort_values("Total_price",ascending=False).head()
import datetime as dt
today_date=dt.datetime(2011,12,10)

rec_df=today_date-df.groupby("Customer ID").agg({"InvoiceDate":max})
rec_df.rename(columns={"InvoiceDate": "Recency"}, inplace=True)
rec_df=rec_df["Recency"].apply(lambda x: x.days)

#FRQUENCY

freq_df=df.groupby("Customer ID").agg({"InvoiceDate":"nunique"})
freq_df.rename(columns={"InvoiceDate": "Frequency"}, inplace=True)

#MONETARY
monetary_df=df.groupby("Customer ID").agg({"Total_price":"sum"})
monetary_df.rename(columns={"Total_price":"Monetary"}, inplace=True)

rfm=pd.concat([rec_df,freq_df, monetary_df], axis=1)
rfm.head()
rfmm=rfm.copy()
#SKEWNESS
rfmm.skew()
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15,3))
sns.distplot(rfmm['Recency'], ax=ax[0])
sns.distplot(rfmm['Frequency'], ax=ax[1])
sns.distplot(rfmm['Monetary'], ax=ax[2]);
# Log transformation

rfmm['Recency']=np.log1p(rfmm['Recency'])
rfmm['Frequency']=np.log1p(rfmm['Frequency'])
rfmm['Monetary']=np.log1p(rfmm['Monetary'])
rfmm.head(3)
# Scaling

from sklearn.preprocessing import Normalizer
transformer = Normalizer().fit(rfmm)
normalized=transformer.transform(rfmm)
normalized_rfm=pd.DataFrame(normalized,columns=rfm.columns)
normalized_rfm.head()
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
#optimum k

from sklearn.cluster import KMeans

sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(normalized_rfm)
    sse[k] = kmeans.inertia_ # SSE to closest cluster centroid

plt.title('The Elbow Method')
plt.xlabel('k')
plt.ylabel('SSE')
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()
!pip install --upgrade pip
!pip install yellowbrick
from yellowbrick.cluster import KElbowVisualizer
kmeans=KMeans()
visu=KElbowVisualizer(kmeans, k=(2,10))
visu.fit(normalized_rfm)
visu.poof()
# K=3 is the optimal choice

k_means = KMeans(n_clusters = 3).fit(normalized_rfm)
segments=k_means.labels_
segments
print(segments.shape)
print(normalized_rfm.shape)
sns.distplot(segments);
normalized_rfm["Segments"] = k_means.labels_
normalized_rfm.head(3)
rfmm.shape
normalized_rfm.shape
# To analyze well

S=pd.DataFrame(normalized_rfm["Segments"])
S=S.reset_index(drop=True)
rfm=rfm.reset_index(drop=True)
df_all=pd.concat([rfm, S], axis=1)
df_all.head(10)
df_all[["Segments","Recency","Frequency","Monetary"]].groupby("Segments").agg(["min","max","mean","count"])