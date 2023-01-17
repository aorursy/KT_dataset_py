import pandas as pd    
import numpy as np
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);
pd.set_option('display.float_format', lambda x: '%.0f' % x)
import matplotlib.pyplot as plt

#Data set reading process was performed.
df_2010_2011 = pd.read_excel("../input/uci-online-retail-ii-data-set/online_retail_II.xlsx", sheet_name = "Year 2010-2011")
df = df_2010_2011.copy()

# DATA PREPROCESSING
# We extract the returned transactions from the data.
returned = df[df["Invoice"].str.contains("C",na=False)].index
df = df.drop(returned, axis = 0)

#How much money has been earned per invoice? (It is necessary to create a new variable by multiplying two variables)
df["TotalPrice"] = df["Quantity"]*df["Price"]
# Missing observations were deleted.
df.dropna(inplace = True)


#RFM METRIC ACCOUNT

#RECENCY METRIC
import datetime as dt
today_date = dt.datetime(2011,12,9)
df["Customer ID"] = df["Customer ID"].astype(int)
temp_df = (today_date - df.groupby("Customer ID").agg({"InvoiceDate":"max"}))
temp_df.rename(columns={"InvoiceDate": "Recency"}, inplace = True)
recency_df = temp_df["Recency"].apply(lambda x: x.days)

# FREQUENCY METRIC
temp_df = df.groupby(["Customer ID","Invoice"]).agg({"Invoice":"count"})
freq_df = temp_df.groupby("Customer ID").agg({"Invoice":"count"})
freq_df.rename(columns={"Invoice": "Frequency"}, inplace = True)

# MONETARY METRIC
monetary_df = df.groupby("Customer ID").agg({"TotalPrice":"sum"})
monetary_df.rename(columns = {"TotalPrice": "Monetary"}, inplace = True)
rfm = pd.concat([recency_df, freq_df, monetary_df],  axis=1)
df = rfm
df["RecencyScore"] = pd.qcut(df['Recency'], 5, labels = [5, 4, 3, 2, 1])
df["FrequencyScore"] = pd.qcut(df['Frequency'].rank(method = "first"), 5, labels = [1,2,3,4,5])
df["MonetaryScore"] = pd.qcut(df['Monetary'], 5, labels = [1,2,3,4,5])
df["RFM_SCORE"] = df['RecencyScore'].astype(str) + df['FrequencyScore'].astype(str) + df['MonetaryScore'].astype(str)
seg_map = {
        r'[1-2][1-2]': 'Hibernating',
        r'[1-2][3-4]': 'At Risk',
        r'[1-2]5': 'Can\'t Loose',
        r'3[1-2]': 'About to Sleep',
        r'33': 'Need Attention',
        r'[3-4][4-5]': 'Loyal Customers',
        r'41': 'Promising',
        r'51': 'New Customers',
        r'[4-5][2-3]': 'Potential Loyalists',
        r'5[4-5]': 'Champions'
}

df['Segment'] = df['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str)
df['Segment'] = df['Segment'].replace(seg_map, regex=True)
df.head()
rfm = df.loc[:,"Recency":"Monetary"]
df.groupby("Customer ID").agg({"Segment": "sum"}).head()
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler((0,1))
df = sc.fit_transform(rfm)
kmeans = KMeans(n_clusters = 10)
k_fit = kmeans.fit(df)
k_fit.labels_
kmeans = KMeans(n_clusters = 10)
k_fit = kmeans.fit(df)
ssd = []

K = range(1,30)

for k in K:
    kmeans = KMeans(n_clusters = k).fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Distance Residual Sums Versus Different k Values")
plt.title("Elbow method for Optimum number of clusters")
from yellowbrick.cluster import KElbowVisualizer
kmeans = KMeans()
visu = KElbowVisualizer(kmeans, k = (2,20))
visu.fit(df)
visu.poof();
kmeans = KMeans(n_clusters = 6).fit(df)
kumeler = kmeans.labels_
pd.DataFrame({"Customer ID": rfm.index, "Kumeler": kumeler})
rfm["cluster_no"] = kumeler
rfm["cluster_no"] = rfm["cluster_no"] + 1
rfm.groupby("cluster_no").agg({"cluster_no":"count"})
rfm.head()