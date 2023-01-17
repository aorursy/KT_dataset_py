import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import RobustScaler, Normalizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.neighbors import LocalOutlierFactor
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.simplefilter(action = "ignore")
#Reading data set and the first 5 observation
data=pd.read_excel('/kaggle/input/uci-online-retail-ii-data-set/online_retail_II.xlsx', sheet_name = "Year 2010-2011")
today_date = dt.datetime(2011,12,30)
data.head()
#Data set consists of 541910 observation units and 8 variables.
data.shape
data.nunique()
#Descriptive statistics of the data set accessed.
data.describe([0.10,0.25,0.50,0.75,0.90,0.99]).T
#How many of which products are there?
data["Description"].value_counts().head()
#What is the most ordered product?
data.groupby("Description").agg({"Quantity":"sum"}).sort_values("Quantity", ascending = False).head()
data["TotalPrice"] = data["Quantity"]*data["Price"]
data.groupby("Invoice").agg({"TotalPrice":"sum"}).head()
#How many orders came from which country?
data["Country"].value_counts().head()
#Which country earned how much?
data.groupby("Country").agg({"TotalPrice":"sum"}).sort_values("TotalPrice", ascending = False).head()
def prepare_rfm_base_dataframe(for_comparison):
    rfm_base_df = data.copy()
    rfm_base_df.dropna(inplace=True) # Customer ID NaN olan gözlemleri drop ediyoruz.

    rfm_base_df["Customer ID"] = rfm_base_df["Customer ID"].astype(int) # Customer ID datatipini değiştiriyoruz.

    rfm_base_df["Monetary"] = rfm_base_df["Quantity"] * rfm_base_df["Price"] # Total_amount degerlerini buluyoruz

    rfm_base_df["Recency"] = (today_date - rfm_base_df["InvoiceDate"])

    rfm_base_df["Recency"] = rfm_base_df["Recency"].apply(lambda x: x.days)

    df_recency = rfm_base_df.groupby(["Customer ID"]).agg({"Recency":max})

    df_frequency = rfm_base_df.groupby(["Customer ID"]).agg({"Invoice":"count"})
    df_frequency.rename(columns={"Invoice": "Frequency"}, inplace = True)

    df_monetary = rfm_base_df.groupby(["Customer ID"]).agg({"Monetary":sum})

    rfm_base_df =  pd.concat([df_recency, df_frequency, df_monetary], axis=1)
    
    if(for_comparison):
        rfm_base_df["RecencyScore"] = pd.qcut(rfm_base_df['Recency'], 5, labels = [5, 4, 3, 2, 1])
        rfm_base_df["FrequencyScore"] = pd.qcut(rfm_base_df['Frequency'].rank(method="first"), 5, labels = [1, 2, 3, 4, 5])
        rfm_base_df["MonetaryScore"] = pd.qcut(rfm_base_df['Monetary'], 5, labels = [1, 2, 3, 4, 5])
        rfm_base_df["RFM_SCORE"] = (rfm_base_df['RecencyScore'].astype(str) 
                                + rfm_base_df['FrequencyScore'].astype(str) 
                                + rfm_base_df['MonetaryScore'].astype(str))
    
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
    
        rfm_base_df['Segment'] = rfm_base_df['RecencyScore'].astype(str) + rfm_base_df['FrequencyScore'].astype(str)
        rfm_base_df['Segment'] = rfm_base_df['Segment'].replace(seg_map, regex=True)

    return rfm_base_df
def modify_for_k_means(rfm_base_df, features):
    # K-Means
    # K-Means'de tranform edilecek degerlerin orjinallerini ayrı birer değişkene atıyoruz.
    rfm_base_df[["Recency_Base","Frequency_Base","Monetary_Base"]] = rfm_base_df[["Recency","Frequency","Monetary"]]
    # log veya root cube transformasyona sokuyoruz. Dağılımı normale yaklaştırabilmek için
    manage_skewness(rfm_base_df, features)
    return rfm_base_df
def manage_skewness(df, columns):
    for column in columns:
        if(df[df[column] < 0].values.size):
            df[column] = np.cbrt(df[column])
        else:
            if(df[df[column] == 0].values.size):
                df[column] = df[column] + 0.1

            df[column] = np.log(df[column])
def run_kmeans(df_rfm, k_means_features, n_clusters):
    normalizer = RobustScaler()
    normalized = normalizer.fit_transform(df_rfm[k_means_features])
    df_rfm[k_means_features] = pd.DataFrame(normalized, columns= k_means_features, index=df_rfm.index)

    model = KMeans(n_clusters=n_clusters)
    model.fit_transform(df_rfm[k_means_features])
    df_rfm["Cluster_No"]  = model.labels_
    df_rfm["Cluster_No"] = df_rfm["Cluster_No"] + 1
# Klasik RFM Analizi
df_rfm = prepare_rfm_base_dataframe(True)
df_rfm[["Segment" ,"Recency","Frequency","Monetary"]].groupby(["Segment"]).agg(["mean","count"])
# K_Means For Comparision
# K-means'de kullanılacak feature listesi
k_means_features_for_comparision = ["Recency","Frequency"]
df_rfm = modify_for_k_means(df_rfm, k_means_features_for_comparision)
kmeans = KMeans()
visu = KElbowVisualizer(kmeans, k = (2,20))
visu.fit(df_rfm[k_means_features_for_comparision])
visu.poof();
# 6 cluster ile modelimizi çalıştıyoruz
run_kmeans(df_rfm, k_means_features_for_comparision, 6)
# Final dataframe'i olusturuyoruz
df_rfm_final = df_rfm[["Recency_Base","Frequency_Base","Monetary_Base","Segment","Cluster_No"]]
df_rfm_final = df_rfm_final.rename(columns={"Recency_Base":"Recency", "Frequency_Base":"Frequency","Monetary_Base":"Monetary"})
df_rfm_final[["Segment", "Cluster_No" ,"Recency","Frequency","Monetary"]].groupby(["Cluster_No","Segment"]).agg(["mean","count"])
# Cluster gorsel
sns.lmplot(data=df_rfm, x="Recency", y="Frequency", hue='Cluster_No', fit_reg=False, legend=True, legend_out=True);
df_rfm_2 = prepare_rfm_base_dataframe(False)
#df_rfm_2.drop(columns=["RecencyScore","FrequencyScore","MonetaryScore","RFM_SCORE","Segment"])
k_means_features_for_individual = ["Recency","Frequency","Monetary"]
df_rfm_2 = modify_for_k_means(df_rfm_2, k_means_features_for_individual)
kmeans = KMeans()
visu = KElbowVisualizer(kmeans, k = (2,20))
visu.fit(df_rfm_2[k_means_features_for_individual])
visu.poof();
# 6 cluster ile modelimizi çalıştıyoruz
run_kmeans(df_rfm_2, k_means_features_for_individual, 5)
# Cluster gorsel
sns.lmplot(data=df_rfm_2, x="Recency", y="Frequency", hue='Cluster_No', fit_reg=False, legend=True, legend_out=True);
df_rfm_final[["Cluster_No", "Monetary"]].groupby(["Cluster_No"]).agg(["mean","count","sum"])
df_rfm_final[["Cluster_No", "Frequency"]].groupby(["Cluster_No"]).agg(["mean","max","min","count"])
df_rfm_final[["Cluster_No", "Recency"]].groupby(["Cluster_No"]).agg(["mean","max","min","count"])