import pandas as pd
import numpy as np
import seaborn as sns
pd.set_option("display.max_columns",None);
pd.set_option("display.max_rows",None);
retail=pd.read_excel("../input/online-retail-ii-dataset/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df=retail.copy()

#DATA PREPROCESSING
df=df[df.notnull().all(axis=1)]
df["Customer ID"]=df["Customer ID"].astype("int64")
df=df[df["Invoice"].astype("str").str.get(0)!="C"]
df=df.drop([125])
df["Total_price"]=df["Quantity"]*df["Price"]
pd.set_option("display.float_format", lambda x: "%.2f" % x)


#RECENCY
import datetime as dt
today_date=dt.datetime(2011,12,10)
rec_df=today_date-df.groupby("Customer ID").agg({"InvoiceDate":max})
rec_df.rename(columns={"InvoiceDate": "Recency"}, inplace=True)
rec_df=rec_df["Recency"].apply(lambda x: x.days)

#FREQUENCY
freq_df=df.groupby("Customer ID").agg({"InvoiceDate":"nunique"})
freq_df.rename(columns={"InvoiceDate": "Frequency"}, inplace=True)

# MONETARY
monetary_df=df.groupby("Customer ID").agg({"Total_price":"sum"})
monetary_df.rename(columns={"Total_price":"Monetary"}, inplace=True)

#RFM
rfm=pd.concat([rec_df,freq_df, monetary_df], axis=1)
rfm["Recency_Score"]= pd.qcut(rfm["Recency"],5, labels=[5,4,3,2,1])
rfm["Frequency_Score"]= pd.qcut(rfm["Frequency"].rank(method="first"),5, labels=[1,2,3,4,5])
rfm["Monetary_Score"]=pd.qcut(rfm["Monetary"],5, labels=[1,2,3,4,5])
rfm["RFM"]=rfm["Recency_Score"].astype(str)+rfm["Frequency_Score"].astype(str)+rfm["Monetary_Score"].astype(str)

#SEG_MAP
seg_map={r'[1-2][1-2]': "Hibernating", r'[1-2][3-4]': "At Risk", r'[1-2]5': "Can't Lose", r'3[1-2]': "About to Sleep",
        r'33': "Need Attention", r'[3-4][4-5]': "Loyal Customers", r'41': "Promising", r'51': "New Customers",
        r'[4-5][2-3]': "Potential Loyalist", r'5[4-5]': "Champions"}

rfm["RFM_Segment"]=rfm["Recency_Score"].astype(str)+ rfm["Frequency_Score"].astype(str)
rfm["RFM_Segment"]=rfm["RFM_Segment"].replace(seg_map,regex=True)
rfm[["RFM_Segment","Recency","Frequency","Monetary"]].groupby("RFM_Segment").agg(["min","max","mean","count"])
rfm.head()
rfmm=rfm.loc[:,"Recency":"Monetary"]

#log(x+1) transformation
rfmm['Recency']=np.log1p(rfmm['Recency'])
rfmm['Frequency']=np.log1p(rfmm['Frequency'])
rfmm['Monetary']=np.log1p(rfmm['Monetary'])

# Scaling

from sklearn.preprocessing import Normalizer
transformer = Normalizer().fit(rfmm)
normalized=transformer.transform(rfmm)
normalized_rfm=pd.DataFrame(normalized,columns=rfmm.columns)

#K-Means
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters = 10).fit(normalized_rfm)
segments=k_means.labels_


normalized_rfm["K-Means_Segment"] = k_means.labels_




S=pd.DataFrame(normalized_rfm["K-Means_Segment"])
S=S.reset_index(drop=True)
rfm2=rfmm.reset_index(drop=True)
df_=pd.concat([rfm2, S], axis=1)
Customer_ID=rfm.reset_index()["Customer ID"]
df_K_Means=pd.concat([Customer_ID,df_], axis=1)
df_K_Means.head(3)
df_K_Means=df_K_Means.reset_index(drop=True)
rfm=rfm.reset_index(drop=True)

df_all=pd.concat([df_K_Means["Customer ID"],rfm.loc[:,"Recency":"Monetary"], rfm["RFM"], rfm["RFM_Segment"],df_K_Means["K-Means_Segment"] ], axis=1)
df_all.head(5)
df_all[["RFM_Segment","Recency","Frequency","Monetary"]].groupby("RFM_Segment").agg(["min","max","mean","count"])
df_all[["K-Means_Segment","Recency","Frequency","Monetary"]].groupby("K-Means_Segment").agg(["min","max","mean","count"])
df_all[(df_all["RFM_Segment"]=="Champions") & (df_all["K-Means_Segment"]==5)].shape
df_all[(df_all["RFM_Segment"]=="Champions") & (df_all["K-Means_Segment"]==5)].head()