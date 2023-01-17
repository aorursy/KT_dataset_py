import pandas as pd

import numpy as np

import seaborn as sns
#Tüm sutünları ve satırları gözlemlemek için kullanılan kod.

pd.set_option('display.max_columns',None); 

pd.set_option("display.max_rows",None);

pd.set_option("display.float_format",lambda x:"%.2f" % x) #ondalık sayılarda virgülden sonra gösterilecek basamak sayısı.
#2010-2011 yıllarındaki veriler okutulmuştur.

df_2010_2011 = pd.read_excel('../input/uci-online-retail-ii-data-set/online_retail_II.xlsx', sheet_name = "Year 2010-2011")
#Veri setinin kopyası alındı. Bundan sonraki aşamalarda bu kopyalanan veri seti kullanılacaktır.

df = df_2010_2011.copy()
#Veri setinin ilk 5 gözlemi

df.head() 
df.info() 
df.shape 
df.describe().T
df.describe([0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99]).T 
#Veri setinde hiç eksik değer var mı?

df.isnull().values.any()
#Veri setindeki her bir değişkenden kaçar tane eksik gözlem var?

df.isnull().sum()
#Veri setindeki eskik (NaN) değerlerden kurtulduk.(Veri seti Customer ID ve Description değişkenlerinde NaN değerlere sahip)

df.dropna(inplace = True)
#Veri setinde hiç eksik değer var mı?

df.isnull().values.any()
#Quantity değişkeninin aykırı değerleri boxplot yöntemiyle gözlemlenmiştir.

import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

sns.boxplot(x = df['Quantity']);
#Price değişkeninin aykırı değerleri boxplot yöntemiyle gözlemlenmiştir.

plt.figure(figsize=(12,6))

sns.boxplot(x=df['Price']);
#Aykırı gözlemde 1. ve 3. çeyreklikler göz önüne alınarak IQR hesaplanır ve alt, üst sınırlar burada belirlenir. Bu sınırların dışında kalan değerler aykırı sayılır. 



for feature in ["Quantity","Price"]:

    Q1=df[feature].quantile(0.01)

    Q3=df[feature].quantile(0.99)

    IQR=Q3-Q1

    upper=Q3+1.5*IQR

    lower=Q1-1.5*IQR

    

    if df[(df[feature] > upper) | (df[feature] < lower)].any(axis=None):

        print(feature,"yes")

    else:

        print(feature,"no")

        

    df = df[~((df[feature] > upper) | (df[feature] < lower))] #Bu kod ile aykırı değerler ele alınmamış silinmiş olarak kabul edilmiştir.



    #Quantity vePrice değişkenlerinde aykırı değer olduğu sonucu ortaya çıkmaktadır. Bu sebeple çıktıda yes yazmaktadır. 
len(df[((df[feature] > upper) | (df[feature] < lower))])
df[df["Invoice"].astype(str).str.startswith("C")].sort_values("Quantity").head()
#Invoice değişkeninde C ile başlayan gözlemler haricindeki gözlemleri al.

df=df[~df["Invoice"].astype(str).str.startswith("C")] 
#Quantity değişkenine göre sırala ve veri setindeki son beş değişkeni erişme

df.sort_values("Quantity",ascending=False).tail() 
#Customer ID'lerin float şeklinde olması hoş gözükmüyor. Integer dönüşümü yapmalıyız.

df["Customer ID"]=df["Customer ID"].astype(int)
df["Description"].nunique()
df["Description"].value_counts().head()
df.groupby("Description").agg({"Quantity":"sum"}).head()  
df.groupby("Description").agg({"Quantity":"sum"}).sort_values("Quantity",ascending=False).head() 
products_incomes = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).iloc[0:20]



plt.figure(figsize=(20,10))

sns.barplot(products_incomes.index, products_incomes.values, palette="Reds_r")

plt.ylabel("Quantity")

plt.title("Best-selling Products");

plt.xticks(rotation=90);
df.groupby("Description").agg({"Quantity":"sum"}).sort_values("Quantity",ascending=False).tail() 
df["Invoice"].nunique()
df["TotalPrice"] = df["Quantity"]*df["Price"]

df.groupby("Invoice").agg({"TotalPrice":"sum"}).head()
df.sort_values("Price",ascending=False).head()
df["Country"].value_counts()
df.groupby("Country").agg({"TotalPrice":"sum"}).sort_values("TotalPrice",ascending=False).head()
df.head()
# Customer ID değişkeni stringe dönüştürülmüştür.

df["Customer ID"]=df["Customer ID"].astype(int)
#Alışverişin yapıldığı ilk gün

df["InvoiceDate"].min()
#Alışverişin yapıldığı son gün

df["InvoiceDate"].max()
# Veri setinin yapısına göre bugünün tarihini belirlemeliyiz. Veri setindeki max tarih analizin yapıldığı gün kabul edilebilir.

import datetime as dt   

today_date=dt.datetime(2011,12,9)

today_date 
#Her bir müşterinin son satın alma tarihleri

df.groupby("Customer ID").agg({"InvoiceDate":"max"}).head() 
# Analizin yapıldığı gün - Son alışveriş yapılan tarih işlemi yapıldığında recency değerini bulduk.

temp_df=(today_date - df.groupby("Customer ID").agg({"InvoiceDate":"max"}))

temp_df.head()
#InvoiceDate sütunun adı Receny olarak değiştirildi.

temp_df.rename(columns={"InvoiceDate":"Recency"},inplace=True)

temp_df.head()
#Her bir değerin sadece günleri alındı.

recency_df=temp_df["Recency"].apply(lambda x:x.days)

recency_df.head()
#Customer ID ve Invoice e göre gruplama yapılıp invoice değişkenin sayısı bulundu.

temp_df=df.groupby(["Customer ID","Invoice"]).agg({"Invoice":"count"})

temp_df.head()
temp_df.groupby("Customer ID").agg({"Invoice":"sum"}).head()
#Invoice sütunun adı Frequency olarak değiştirildi.

freq_df=temp_df=df.groupby("Customer ID").agg({"Invoice":"count"})

freq_df.rename(columns={"Invoice":"Frequency"},inplace=True)

freq_df.head()
df.head()
#Customer ID e göre gruplandırıp TotalPrice toplamını bulduk.

monetary_df=df.groupby(["Customer ID"]).agg({"TotalPrice":"sum"})

monetary_df.head()
#TotalPrice sütunun adı Monetary olarak değiştirilmiştir.

monetary_df.rename(columns={"TotalPrice":"Monetary"},inplace=True)

monetary_df.head()
print(recency_df.shape,freq_df.shape,monetary_df.shape)
# Hepsinde customer ID ortak lduğunda customer ID yi index algıladı

rfm = pd.concat([recency_df,freq_df,monetary_df], axis=1)

rfm.head(10)
# "qcut" quartile değerlerine göre bölme yapar.

rfm["RecencyScore"]=pd.qcut(rfm["Recency"],5,labels=[5,4,3,2,1])  #Recency de küçük olan iyiyken monetary ve frequency de büyük olanlar iyidir.
rfm["FrequencyScore"]=pd.qcut(rfm["Frequency"],5,labels=[1,2,3,4,5])
rfm["MonetaryScore"]=pd.qcut(rfm["Monetary"],5,labels=[1,2,3,4,5])
rfm.head()
(rfm["RecencyScore"].astype(str)+ rfm["FrequencyScore"].astype(str)+ rfm["MonetaryScore"].astype(str)).head()
rfm["RFM_SCORE"]=rfm["RecencyScore"].astype(str)+rfm["FrequencyScore"].astype(str)+rfm["MonetaryScore"].astype(str)
#Tabloda en iyi olan müşteriler (Champions)

rfm[rfm["RFM_SCORE"]=="555"].head()
#Tabloda en kötü müşteriler

rfm[rfm["RFM_SCORE"]=="111"].head()
#en iyi ve en kötü müşterilerin doğruluğuna bakabiliriz.

rfm.describe().T
#Burada rfm skorlarına göre segmentler atanmaktadır. 

#Örnek olarak Hibernating sınıfı şu skor değerlerine sahiptir. 

#r'[1-2][1-2]':'Hibernating' = İlk bölüm R yi ikinci kısım F yi ifade etmektedir. R de 1-2, F de 1-2 görürsen Hibernating yaz demek. 

#Buraya sadece Recency ve Frequency eklenmesinin sebebi tabloda sadece bu iki parametre yer aldığı için ancak Monetary de yanlarına eklenebilir.



seg_map={

    r'[1-2][1-2]':'Hibernating',

    r'[1-2][3-4]':'At Risk',

    r'[1-2]5':'Can \t Loose',

    r'33':'Need Attention',

    r'[3-4][4-5]':'Loyal Customers',

    r'41':'Promising',

    r'51':'New Customers',

    r'[4-5][2-3]':'Potential Loyalists',

    r'5[4-5]':'Champions'

}
rfm['Segment']=rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str)

rfm['Segment']=rfm['Segment'].replace(seg_map,regex=True)

rfm.head()
#Segmentler %kaçlık kısmı oluşturmakta.

segments_counts = rfm['Segment'].value_counts().sort_values(ascending=True)



fig, ax = plt.subplots()



bars = ax.barh(range(len(segments_counts)),

              segments_counts,

              color='silver')

ax.set_frame_on(False)

ax.tick_params(left=False,

               bottom=False,

               labelbottom=False)

ax.set_yticks(range(len(segments_counts)))

ax.set_yticklabels(segments_counts.index)



for i, bar in enumerate(bars):

        value = bar.get_width()

        if segments_counts.index[i] in ['Can\'t loose']:

            bar.set_color('firebrick')

        ax.text(value,

                bar.get_y() + bar.get_height()/2,

                '{:,} ({:}%)'.format(int(value),

                                   int(value*100/segments_counts.sum())),

                va='center',

                ha='left'

               )



plt.show()
#Yeni dataFrame oluşturup Customer ID leri atadık.

new_df=pd.DataFrame()

new_df["NewCustomerID"]=rfm[rfm["Segment"]=="New Customers"].index
new_df.head()
#Hazırladığımız müşteriyi yeni müşteriler adında excel olarak kaydettik.Bu dosya müşteri departmanı ile paylaşılmaya hazır hale gelmiştir.

new_df.to_csv("new_customers.csv")
rfm[["Segment","Recency","Frequency","Monetary"]].groupby("Segment").agg(["mean","count"])