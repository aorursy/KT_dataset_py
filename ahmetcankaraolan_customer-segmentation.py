#Gerekli olan kütüphanelerin import işlemleri yapılmıştır.
import pandas as pd
import numpy as np
import seaborn as sns

#Tüm sutünları ve satırları gözlemlemek için kullandığımız kod.
pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);

#virgulden sonra gösterilecek olan sayı sayısı
pd.set_option('display.float_format', lambda x: '%.0f' % x)
import matplotlib.pyplot as plt
#Veri seti okuma işlemi gerçekleştirildi.
df_2010_2011 = pd.read_excel("../input/uci-online-retail-ii-data-set/online_retail_II.xlsx", sheet_name = "Year 2010-2011")
#Veri seti boyutu büyük olduğu için okuma işlemi uzun sürdü bu sebeple veri setini kopyalama işlemi yaptık.
df = df_2010_2011.copy()
#Veri setine ait ilk 5 gözlem birimine eriştik.
df.head()
#essiz ürün sayısı nedir?
df["Description"].nunique()
#hangi üründen kaçar tane var?
df["Description"].value_counts().head()
#en çok sipariş edilen ürün hangisi?
df.groupby("Description").agg({"Quantity":"sum"}).head()
#yukarıdaki çıktıyı nasıl sıralarız?
df.groupby("Description").agg({"Quantity":"sum"}).sort_values("Quantity", ascending = False).head()
#toplam kaç fatura kesilmiştir?
df["Invoice"].nunique()
#fatura basina toplam kac para kazanilmistir? (iki değişkeni çarparak yeni bir değişken oluşturmak gerekmektedir)
#kaç para kazanıldığını bulmak için "adet(Quantity) * fiyat(Price)" yapılması gerekir. 
df["TotalPrice"] = df["Quantity"]*df["Price"]
#Dataframe "Total Price" değişkeni eklenmiştir. Bu değişken ilk 5 gözlem üzerinden incelenmiştir. 
df.head()
#Fatura numarasına (Invoice) göre gruplayıp "Total Price" değişkenin toplamını getirildi. 
#Bu şekilde fatura başına toplam kaç para kazanıldı hesaplanmıştır.
df.groupby("Invoice").agg({"TotalPrice":"sum"}).head()
#en pahalı ürünler hangileri?
#Ürünlerin ismine göre groupby yapılarak "Price" değişkenin max değeri alınmıştır ve azalan şekilde sıralanmıştır.
df.groupby("Description").agg({"Price":"max"}).sort_values("Price", ascending = False).head()
#en pahalı ürünler hangileri? 
#Farklı bir çözüm olarak direk dataframe üzerinden "Price" değişkenine göre azalan şekilde sıralama yapılmaktadır. 
df.sort_values("Price", ascending = False).head()
#hangi ülkeden kac sipariş geldi?
#Ülke kategorik bir değişken olduğu için kategorik değişkenlerin sınıflarını value_counts() fonksiyonu ile saydırabiliriz.
df["Country"].value_counts().head()
#hangi ülke ne kadar kazandırdı?
#Ülkelere göre groupby yapılarak "Total Price" değişkenini toplamı alınarak hangi ülke ne kadar bırakmış öğrenilmiştir.
df.groupby("Country").agg({"TotalPrice":"sum"}).sort_values("TotalPrice", ascending = False).head()
#en çok iade alan ürün hangisidir?
#Invoice değişkeninde yer alan kodlar başında "C" ifadesi iade anlamına gelmektedir.
#Veri setinde şu anda na değerler olduğu için str.contains komutu içinde na değerlerini False olarak tanımlayıp görmezden gel diyoruz.
#Adet işlemi için de "Quantity" değişkenini azalan şekilde sıralarsak yakaladığımız Invoice değişkenine göre en çok iadeleri buluruz.
returned = df[df["Invoice"].str.contains("C",na=False)]
returned.sort_values("Quantity", ascending = True).head()
# Hiç eksik gözlem var mı sorusunu sormaktadır.
df.isnull().sum()
# Elimdeki veri setinde yaklaşık olarak 500000 gözlemden oluşan bir veri seti var bu nedenle eksik gözlemleri silebilirim.
df.dropna(inplace = True)
#Silme işleminden sonra tekrardan kontrol ediyorum eksik gözlem var mı?
df.isnull().sum()
#Boyut bilgisine erişilmektedir.
df.shape
#Çeyrekliklerini(Kartiller) kendimiz belirleyerek betimsel istatistiklere bakılmaktadır. Veri seti hakkında incelemeler yapılmaktadır.
df.describe([0.05,0.01,0.25,0.50,0.75,0.80,0.90,0.95,0.99]).T
# Aykırı gözlem analizinde en yaygın kullanılan method bir alt limit ve üst limit belirleyerek baskılamaktır.
# Burada 1. ve 3. çeyreklikler göz önüne alınarak IQR hesaplanır ve alt, üst sınırlar burada belirlenir. 
# Neden sadece bu değişkenler çünkü yukarıdaki kodda Customer ID hariç aykırı gözlem analizi olduğu saptanmıştır.
#Burada aykırı gözlemlere ucundan değinilerek bu problem ortadan kaldırılmıştır.
for feature in ["Quantity","Price","TotalPrice"]:

    Q1 = df[feature].quantile(0.01)
    Q3 = df[feature].quantile(0.99)
    IQR = Q3-Q1
    upper = Q3 + 1.5*IQR
    lower = Q1 - 1.5*IQR

    if df[(df[feature] > upper) | (df[feature] < lower)].any(axis=None):
        print(feature,"yes")
        print(df[(df[feature] > upper) | (df[feature] < lower)].shape[0])
    else:
        print(feature, "no")
df.head()
df.info()
df.shape
#Alışverişin yapıldığı ilk gün
df["InvoiceDate"].min()
#Alışverişin yapıldığı son gün
df["InvoiceDate"].max()
#Analiz için gerekli import işlemi yapılarak max değer analizi yaptığımız gün olarak belirlenmiştir. Bu değer today_date olarak atanmıştır.
import datetime as dt
today_date = dt.datetime(2011,12,9)
today_date
#Customer ID değerine göre groupby işlemi yapılıp Invoice Date değişkenin max değerleri getirildi. 
#Yani son alışveriş yapılan tarihler geldi. Müşteri ID sine göre son yapılan alışveriş tarihlerine ulaştık. 
df.groupby("Customer ID").agg({"InvoiceDate":"max"}).head()
# Customer ID değişkeni stringe dönüştürülmüştür.
df["Customer ID"] = df["Customer ID"].astype(int)
# Analizin yapıldığı gün - Son alışveriş yapılan tarih işlemi yapıldığında recency değerini yakaladık.
temp_df = (today_date - df.groupby("Customer ID").agg({"InvoiceDate":"max"}))
temp_df.head()
# Analiz yapıldı ve burada InvoıceDate değişkeninin adını Recency olarak değiştirilmektedir.
temp_df.rename(columns={"InvoiceDate": "Recency"}, inplace = True)
temp_df.head()
#Yukarıda yapılan işlem sonucunda Recency değişkeninde saat değerlerini silmek istiyorum.
#Bunu yapabilmek için, daha önce burası Invoice Date değişkeni ve veri tipi datetime olduğu için days'leri seçip saatleri silebiliriz. 
recency_df = temp_df["Recency"].apply(lambda x: x.days)
recency_df.head()
#Customer ID ye göre ve Invoice'a gruopby yap ve Invoice değişkenini say.
temp_df = df.groupby(["Customer ID","Invoice"]).agg({"Invoice":"count"})
temp_df.head()
#Customer ID ye göre gruopby yap ve Invoice değişkenini topla. Yani her müşteri kaç alışveriş yapmış toplamda bunu bulmuş oluyoruz.
temp_df.groupby("Customer ID").agg({"Invoice":"count"}).head()
#Burada yukarıda yapmış olduğumuz işlemi bir değişkene atayıp "Invoice" yazan değişkeni de "Frequency" olarak değiştirip gözlemliyoruz.
freq_df = temp_df.groupby("Customer ID").agg({"Invoice":"count"})
freq_df.rename(columns={"Invoice": "Frequency"}, inplace = True)
freq_df.head()
monetary_df = df.groupby("Customer ID").agg({"TotalPrice":"sum"})
monetary_df.head()
#Sutün adlandırması değiştirildi.
monetary_df.rename(columns = {"TotalPrice": "Monetary"}, inplace = True)
monetary_df.head()
# Her bir parametrenin boyutlarına bakıldı. 
print(recency_df.shape,freq_df.shape,monetary_df.shape)
#Parametreleri bir dataframe olarak concat fonksiyonu yardımı ile birleştirdik.
rfm = pd.concat([recency_df, freq_df, monetary_df],  axis=1)
rfm.head()
# Buradaki skorlama işlemi pandas içinde .qcut fonk. kullanılarak bölünmüştür.
rfm["RecencyScore"] = pd.qcut(rfm['Recency'], 5, labels = [5, 4, 3, 2, 1])
rfm["FrequencyScore"] = pd.qcut(rfm['Frequency'].rank(method = "first"), 5, labels = [1,2,3,4,5])
rfm["MonetaryScore"] = pd.qcut(rfm['Monetary'], 5, labels = [1,2,3,4,5])
#Skorlama işlemi sonrası dataframe ilk 5 gözlem biriminin incelenmesi
rfm.head()
#Skorları yanyana yazdırmak için yapılmaktadır.
(rfm['RecencyScore'].astype(str) + 
 rfm['FrequencyScore'].astype(str) + 
 rfm['MonetaryScore'].astype(str)).head()
#Dataframe e RFM_SCORE sutünü eklenmiştir.
rfm["RFM_SCORE"] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str) + rfm['MonetaryScore'].astype(str)
rfm.head()
#Betimsel istatistikleri incelenerek bazı yorumlar yapılmaktadır.
rfm.describe().T
#En iyi müşteriler gösterilmektedir.
rfm[rfm["RFM_SCORE"] == "555"].head()
#En kötü müşteriler gösterilmektedir.
rfm[rfm["RFM_SCORE"] == "111"].head()
#Burada rfm skorlarına göre sınıflar atanmaktadır. Bu sınıfların hangi skor aralığında olacağı aşağıda belirtilmiştir. 
#Örnek olarak Hibernating sınıfı şu skor değerlerine sahiptir. 
#Recency değeri 1-2, Frequency değeri 1-2 olanlardır diğer sınıflarda bu şekilde okunmaktadır.
#Buraya sadece Recency ve Frequency eklenmesinin sebebi tabloda sadece bu iki parametre yer aldığı için ancak Monetary de yanlarına eklenebilir.
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
#Yukarıda tanımlanan seg_map'i dataframe dahil etme işlemi yapılmıştır.
rfm['Segment'] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str)
rfm['Segment'] = rfm['Segment'].replace(seg_map, regex=True)
rfm.head()
#Segmentlere göre groupby yaparak elde edilen (kodda belirtilen) 3 parametrenin ortalama ve kaç adet olduğunu getirmektedir. 
rfm[["Segment", "Recency","Frequency","Monetary"]].groupby("Segment").agg(["mean","count"]).head()
#Bu segmentteki grup dikkat gerektiren grublardan birisiydi. Bunları yakalama işlemi yapılmıştır.
rfm[rfm["Segment"] == "Need Attention"].head()
#Yeni müşterilere ait Customer ID (indeks) değerlerine erişilmiştir.
#Bu değerler müşteri departmanı ile paylaşılıp yeni müşterilere özel promosyonlar ve mailler bu ID'ler sayesinde atılabilir.
rfm[rfm["Segment"] == "New Customers"].index
#Dikkat gerektiren müşterilerin Customer ID (indeks) değerlerine erişilmiştir.
#Bu değerler müşteri departmanı ile paylaşılıp dikkat gerektiren müşterilere özel promosyonlar ve mailler bu ID'ler sayesinde atılabilir.
rfm[rfm["Segment"] == "Need Attention"].index
#Yeni bir dataframe oluşturularak içerisine Need Attention grubuna ait müşterilerin ID bilgileri atılmıştır.
new_df = pd.DataFrame()
new_df["NeedAttentionID"] = rfm[rfm["Segment"] == "Need Attention"].index
#İlk 5 gözlem gözlemlenmiştir.
new_df.head()
#Atanan bu ID'ler excel çıktısı alınarak müşteri departmanı ile paylaşılmaya hazır hale getirilmiştir.
new_df.to_excel("new_attention.xlsx")
#Hangi segmentten kaç adet bulunmakta ve %kaçını oluşturmakta.
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
rfm[["Segment", "Recency","Frequency","Monetary"]].groupby("Segment").agg(["mean","count","max"]).head(20)