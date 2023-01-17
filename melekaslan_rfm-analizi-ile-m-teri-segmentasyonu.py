import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);
pd.set_option('display.float_format', lambda x: '%.0f' % x)
df_2010_2011 = pd.read_excel("online_retail_II.xlsx", sheet_name = "Year 2010-2011")
df = df_2010_2011.copy()
#Hangi üründen kaç tane var?
df["Description"].value_counts().head()
#Eşsiz ürün sayısı kaçtır?
df["Description"].nunique()
#En çok sipariş edilen ürün hangisidir?
df.groupby("Description").agg({"Quantity":"sum"}).sort_values("Quantity", ascending = False).head()
#Toplam kaç fatura kesilmiştir?
df["Invoice"].nunique()
#Fatura başına ortalama kaç para kazanılmıştır?
df["TotalPrice"] = df["Quantity"]*df["Price"]
df.groupby("Invoice").agg({"TotalPrice":"sum"}).head()
df.head()
#En pahalı ürünler hangileridir?
df.sort_values("Price", ascending = False).head()
#Hangi ülkeden kaç sipariş geldi?
df["Country"].value_counts()
#Hangi ülke ne kadar kazandırdı?
df.groupby("Country").agg({"TotalPrice":"sum"}).sort_values("TotalPrice", ascending = False).head()
#En çok iade alan ürün hangisidir? 
#Invoice değişkeninde başında "C" ifadesi yer alan ürünler iptal edilmiştir. Veri setinde şu anda boş yani değerler bulunmaktadır.
#Bu yüzden str.contains komutu içinde na değerlerini False olarak tanımlayıp görmezden geliyoruz. 
#Adet işlemi için de "Quantity" değişkenini azalan şekilde sıralayarak en çoktan en aza doğru iadeleri sıralayabiliriz.
return_ = df[df["Invoice"].str.contains("C", na=False)]
return_.sort_values("Quantity", ascending = True).head(3)
#Eksik değer olup olmadığına bakıyoruz.
df.isnull().sum()
df.shape
#Eksik değerleri veri setinden çıkarıyoruz.
df.dropna(inplace = True)
df.shape
#İstatistiksel değerleri, kartilleri inceleyerek veri seti hakkında bilgi ediniyoruz.
df.describe([0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95, 0.99]).T
for feature in ["Quantity","Price","TotalPrice"]:

    Q1 = df[feature].quantile(0.01)
    Q3 = df[feature].quantile(0.99)
    IQR = Q3-Q1
    upper = Q3 + 1.5*IQR
    lower = Q1 - 1.5*IQR

    if df[(df[feature] > upper) | (df[feature] < lower)].any(axis=None):
        print(feature,"→ YES")
        print(df[(df[feature] > upper) | (df[feature] < lower)].shape[0])
    else:
        print(feature, "→ NO")
df.info()
#Alışveriş yapılan ilk günü buluyoruz.
df["InvoiceDate"].min()
#Alışveriş yapılan son günü buluyoruz.
df["InvoiceDate"].max()
#İnceleyeceğimiz veriler 2010-2011 yılları arasında olduğu için bugünün tarihini kendimiz belirliyoruz.
import datetime as dt
today_date = dt.datetime(2011,12,9)
today_date
#Müşteri ID'sine göre son yapılan alışveriş tarihlerine ulaşıyoruz.
df.groupby("Customer ID").agg({"InvoiceDate":"max"}).head()
#(Bizim belirlediğimiz bugünün tarihi - alışveriş yapılan son tarih) işlemi yapıldığında recency değerini buluyoruz.
temp_df = (today_date - df.groupby("Customer ID").agg({"InvoiceDate":"max"}))
temp_df.rename(columns={"InvoiceDate": "Recency"}, inplace = True)
temp_df.head()
#Saat değerlerini silmek için, daha önce burası Invoice Date değişkeni ve veri tipi datetime olduğu için days'leri seçip saatleri siliyoruz.
recency_df = temp_df["Recency"].apply(lambda x: x.days)
recency_df.head()
temp_df = df.groupby(["Customer ID","Invoice"]).agg({"Invoice":"count"})
temp_df.head()
#Her müşteri kaç alışveriş yapmış toplamda bunu buluyoruz.
temp_df.groupby("Customer ID").agg({"Invoice":"count"}).head()
#Yukarıda yapmış olduğumuz işlemi bir değişkene atayıp "Invoice" yazan değişkeni de "Frequency" olarak değiştirip gözlemliyoruz.
freq_df = temp_df.groupby("Customer ID").agg({"Invoice":"count"})
freq_df.rename(columns={"Invoice": "Frequency"}, inplace = True)
freq_df.head()
monetary_df = df.groupby("Customer ID").agg({"TotalPrice":"sum"})
monetary_df.head()
#Sutün adlandırmasını değiştiriyoruz.
monetary_df.rename(columns={"TotalPrice": "Monetary"}, inplace = True)
monetary_df.head()
print(recency_df.shape,freq_df.shape,monetary_df.shape)
#Parametreleri bir dataframe olarak concat fonksiyonu yardımı ile birleştiriyoruz.
rfm = pd.concat([recency_df, freq_df, monetary_df],  axis=1)
rfm.head()
#Değerler görüldüğü üzere 1 ile 5 arasında olan değerlerdir. 5 çok iyi, 1 ise çok kötü anlamına gelmektedir.
rfm["RecencyScore"] = pd.qcut(rfm['Recency'], 5, labels = [5, 4, 3, 2, 1])
rfm["FrequencyScore"] = pd.qcut(rfm['Frequency'].rank(method = "first"), 5, labels = [1,2,3,4,5])
rfm["MonetaryScore"] = pd.qcut(rfm['Monetary'], 5, labels = [1, 2, 3, 4, 5])
rfm.head()
#Skorları yanyana yazdırıyoruz.
(rfm['RecencyScore'].astype(str) + 
 rfm['FrequencyScore'].astype(str) + 
 rfm['MonetaryScore'].astype(str)).head()
#RFM_SCORE sütununu Dataframe tablosuna ekliyoruz.
rfm["RFM_SCORE"] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str) + rfm['MonetaryScore'].astype(str)
rfm.head()
#Betimsel istatistikleri inceliyoruz.
rfm.describe().T
#En iyi müşteriler gösterilmektedir.
rfm[rfm["RFM_SCORE"] == "555"].head()
#En kötü müşteriler gösterilmektedir.
rfm[rfm["RFM_SCORE"] == "111"].head()
#RFM skorlarına göre müşterileri sınıflara atıyoruz. Bu sınıfların hangi skor aralığında olacağı aşağıda belirtilmiştir. 
#Örnek olarak Hibernating sınıfı şu skor değerlerine sahiptir. 
#Recency değeri 1-2, Frequency değeri 1-2 olanlardır diğer sınıflarda bu şekilde okunmaktadır.
#Buraya sadece Recency ve Frequency eklenmesinin sebebi tabloda sadece bu iki parametre yer almaktadır.
#Ancak Monetary de eklenebilir.
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
#Yukarıda tanımlanan seg_map'i dataframe tablosuna dahil ediyoruz.
rfm['Segment'] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str)
rfm['Segment'] = rfm['Segment'].replace(seg_map, regex=True)
rfm.head()
#Segmentlere göre groupby yaparak elde edilen 3 parametrenin ortalama ve kaç adet olduğunu buluyoruz.
rfm[["Segment", "Recency","Frequency","Monetary"]].groupby("Segment").agg(["mean","count"])
#"Need Attention" grubunu yakalıyoruz.
rfm[rfm["Segment"] == "Need Attention"].head()
#Dikkat gerektiren müşterilerin Customer ID (indeks) değerlerine erişilmiştir.
#Yeni müşterilere özel promosyonlar ve mailler bu ID'ler sayesinde atılabilir.
rfm[rfm["Segment"] == "Need Attention"].index
#Yeni bir dataframe oluşturularak içerisine Need Attention grubuna ait müşterilerin ID bilgileri atıyoruz.
new_df = pd.DataFrame()
new_df["NewCustomerID"] = rfm[rfm["Segment"] == "Need Attention"].index
new_df.head()
#Atanan bu ID'ler csv çıktısı alınarak kullanıma hazır hale getirilmiştir.
new_df.to_csv("new_customers.csv")
rfm[["Segment", "Recency","Frequency","Monetary"]].groupby("Segment").agg(["mean","count","max"]).head(10)

