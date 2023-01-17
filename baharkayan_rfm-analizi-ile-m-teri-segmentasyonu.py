#Warningleri çalışmamda görmek istemediğim için kapatıyorum.

import warnings

warnings.simplefilter(action='ignore')





import pandas as pd

#virgulden sonra gösterilecek olan sayı 2 basamak olarak ayarlanıyor.

pd.set_option('display.float_format', lambda x: '%.2f' % x)



#Çıktılarda tüm satır ve sütunları görebilmek için kütüphane ayarı yapılıyor.

pd.set_option('display.max_columns', None); 

pd.set_option('display.max_rows', None);



#Excel okuma işlemini gerçekleştiriyorum. 2.Sheet olan "Year 2010-2011" üzerinden çalışmamı gerçekleştireceğim.

#"df" isimli Dataframe yapımı oluşturuyorum.



df_2010_2011 = pd.read_excel("../input/uci-online-retail-ii-data-set/online_retail_II.xlsx", sheet_name = "Year 2010-2011")

df = df_2010_2011.copy()

#İade faturalarım için df_iade isimli dataframe yapımı oluşturuyorum. "Invoice" kolonu "C" ile başlayanları bu dataframe'e atıyorum.

df_iade = df[df["Invoice"].str.startswith("C", na = False)]

#Orjinal dataframe yapımdan iadeleri çıkartarak, satış verilerini çekiyorum, df_satis olarak kaydediyorum.



df_satis = df[~(df["Invoice"].str.startswith("C", na = False))]

# Satış veri setinde NaN kontrolü yapıyorum. Değişken bazında NaN olan sayılara bakacağım.

# "Description" için 1454 değerin, "Customer ID" için 134697 değerin NaN olduğunu gözlemledim.

df_satis.isnull().sum()
#Customer ID değerleri NaN olanlar, RFM Analizinde işime yaramayacağı için uçuruyorum.

df_satis.dropna(subset=['Customer ID'], how='all', inplace=True)
#Satış veri setinde hiç NaN veri kalmadığını gözlemledim. 

#Buradan, Description değeri NaN olan tüm kayıtların, Customer ID değerlerinin de NaN olduğu bilgisini çıkarttım.

df_satis.isnull().sum()
#Satış veri setindeki silinen NaN değerlerinin sayısını tutuyorum. Aşağıda doğrulama için kullanacağım.

satis_nan = 134697
# İade veri setinde NaN kontrolü yapıyorum. Değişken bazında NaN olan sayılara bakacağım.

# "Customer ID" için 383 değerin NaN olduğunu gözlemledim.

df_iade.isnull().sum()
#Customer ID değerleri NaN olanları müşteri bazında yorumlayamayacağım ve işime yaramayacağı için uçuruyorum.

df_iade.dropna(subset=['Customer ID'], how='all', inplace=True)
#İade veri setinde hiç NaN veri kalmadığını gözlemledim. 

df_iade.isnull().sum()
#İade veri setindeki silinen NaN değerlerinin sayısını tutuyorum. Aşağıda doğrulama için kullanacağım.

iade_nan = 383
#Satış veri setinin ilk 5 gözlemi

df_satis.head()
# df_satis HAKKINDA YAPISAL BİLGİLER : 

# Burada  Customer ID ve Price değişkenlerinin float tipinde olduğunu görüyorum. Kütüphane Ayarları bölümünde, float değişkenler için "virgülden sonra 2 basamak gösterilsin" ayarı yapmıştım.

# Bu durum Customer ID yi de etkiledi. Yukarıda da "Cutomer ID" için virgülden sonra 2 basamak yazıldığını gözlemledim.

# "Customer ID" aslında kategorik değişken olmalı. Bu değişkene tip dönüşümü uygulayacağım.

df_satis.info()
# Customer ID TİP DÖNÜŞÜMÜ

# Hem satış veri seti, hem de iade veri seti için bu dönüşümü uyguluyorum. 

# İlk önce integera çevirerek, virgülden sonraki basamaklardan kurtulacağım. Sonrasında string tipine dönüştürerek, kategorik değişken yapacağım.

df_satis["Customer ID"] = df_satis["Customer ID"].astype(int)

df_iade["Customer ID"] = df_iade["Customer ID"].astype(int)

#Hem satış veri setinde, hem de iade veri setinde "Customer ID" kolonunu kategorik deişkene çevirmek için string dönüşümü uyguluyorum.

df_satis["Customer ID"] = df_satis["Customer ID"].astype(str)

df_iade["Customer ID"] = df_iade["Customer ID"].astype(str)
#Satış veri setimin yapısal bilgilerine yeniden bakıyorum.

#Customer ID nin kategorik değişkene dönüştüğünü gözlemledim.(object type)

df_satis.info()
#Customer ID nin virgülden sonraki basamaklarının uçtuğunu gözlemliyorum. İşlemimiz başarılı.

df_satis.head()
#Orjinal veri setinin boyut bilgisi

df_2010_2011.shape
# Satış veri seti boyut bilgisi

df_satis.shape
# İade veri seti boyut bilgisi 

df_iade.shape
# Satış ve İade veri setlerinden silinen NaN kayıtların toplamı

top_nan = satis_nan + iade_nan

top_nan
# Yukarıda verilen tanımlara göre, ana toplamın tutup tutmadığı kontrolü yapılıyor.

#True döndüğü için verilerde bir kaybımız olmadı. Yaptığımız işlemleri veri sayısı bakımından doğrulamış olduk

df_2010_2011.shape[0] == df_satis.shape[0] + df_iade.shape[0] + top_nan
#df_satis kolon isimleri

df_satis.columns
# df_satis DEĞİŞKENLERİNİN İSTATİSTİKSEL BİLGİLERİ 

# Buraya sadece sayısal değişkenlerin bilgileri gelir.

df_satis.describe().T
#Satış veri setindeki tüm satılan ürünlerin sayısı(Eşsiz ürün kodu toplamı)

df_satis["StockCode"].nunique()
#İade Verisindeki tüm ürünlerin sayısı(Eşsiz ürün kodu toplamı)

df_iade["StockCode"].nunique()
#Satış Verisinde hangi üründen kaçar tane var? İlk 5 gözlemi getir.

df_satis["StockCode"].value_counts().head()
#Satış verisinde en çok sipariş edilen ürünler hangileri? İlk 5 gözlemi getir.

df_satis.groupby("StockCode").agg({"Quantity":"sum"}).sort_values("Quantity", ascending = False).head()
#İade verisinde en çok iade edilen ürünler hangileri? İlk 5 gözlemi getir.

#Buradaki negatiflik, ürünün iade olduğu anlamına gelir. Vektörel bir anlam gibi düşünülebilir. 

# Sayısal olarak -1 büyük olmasına rağmen, miktarsal olarak bir büyüklük ifade etmeyeceğinden, satış verisindekinin aksine, "ascending" değerini burada "True" vererek, en çok iade alanı en başta görebiliriz.

df_iade.groupby("StockCode").agg({"Quantity":"sum"}).sort_values("Quantity", ascending = True).head()
#Satış veri setinde toplam kaç fatura kesilmiştir?

df_satis["Invoice"].nunique()
#Satış veri setinde her bir kayıt için toplam fiyat hesaplayalım ve yeni bir kolon olarak ekleyelim.

df_satis["TotalPrice"] = df_satis["Quantity"]*df_satis["Price"]

df_satis.head()
#Satış veri setinde fatura başına ne kadar para kazanılmıştır? En fazla para kazanılan faturalar hangileridir? İlk 5 gözlemi getirelim.

df_satis.groupby("Invoice").agg({"TotalPrice":"sum"}).sort_values("TotalPrice", ascending = False).head()
#Satış veri setinde en pahalı ürünler hangileridir? İlk 5 gözlemi getirelim.

df_satis.sort_values("Price",ascending = False).head()
#Satış veri setinde hangi ülkeden kaç sipariş geldi?

df_satis.groupby("Country").agg({"Quantity":"sum"}).sort_values("Quantity",ascending = False)
#Satış veri setindeki, ülkelerden alınan maximum sipariş miktarlı ürün kodlarını, Ülke, Ürün Kodu ve Miktar bazında alalım



df_sonuc = df_satis.groupby(["Country","StockCode"]).agg({"Quantity":"sum"}).sort_values("Quantity", ascending = False)

df_sonuc = df_sonuc.reset_index()

df_sonuc.groupby("Country").head(1)
#Satış veri setinde hangi ülke ne kadar kazandırdı?

df_satis.groupby("Country").agg({"TotalPrice":"sum"}).sort_values("TotalPrice",ascending = False)
#Satış veri setinin istatistiksel bilgilerini daha detaylı gözlemleyelim.

df_satis.describe([0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95, 0.99]).T
#Veri seti eski tarihli bir veri seti olduğu için, müşterilerin son temasından bugüne kadar olan süreyi hesaplarsak, bu süre çok uzun olacaktır.

#Biz kendimize örnek olması açısından, veri seti üzerinden bir "bugün" belirleyelim. Bunu da veri seti üzerindeki en son tarihli gün olarak alalım.



max_date = df_satis["InvoiceDate"].max()

max_date
#max_date değişkenini datetime tipi olarak almalıyız. Böylece günler arasında çıkartma işlemini yapabileceğiz.

import datetime as dt

bugün = dt.datetime(2011,12,9,12,50,0)

bugün
#Bugünün tarihinden, müşterilerin son alışveriş tarihlerini çıkartarak arada geçen sürenin gün değerlerini yeni bir dataframe yapısına atalım. Bu değeler müşterinin "Recency" değerleridir. İlk 5 gözleme bakalım

df_tempr = (bugün - df_satis.groupby("Customer ID").agg({"InvoiceDate":"max"})).rename(columns={"InvoiceDate":"Recency"})["Recency"].apply(lambda x : x.days)

df_recency = pd.DataFrame(df_tempr)

df_recency.head()
# Her bir müşteriye kesilmiş farklı fatura sayılarını bulursak, müşterinin toplam satın alma sayısını bulmuş oluruz. Bu da frequency değerimizdir.



#İlk önce Müşteri ID ve Fatura No bazında gruplayarak, her bir faturanın Müşteri bazında kaç sefer çoklandığını gözlemliyoruz.İlk 5 gözleme bakalım

df_tempf = df_satis.groupby(["Customer ID","Invoice"]).agg({"Invoice":"count"})

df_tempf.head()
#Şimdi de df_tempf üzerinde her bir müşterideki fatura sayısını saydırmamız lazım. Bulduğumuz değerler, müşterilerin "Frequency" değerleridir. İlk 5 gözleme bakalım

df_frequency = df_tempf.groupby("Customer ID").agg({"Invoice":"count"}).rename(columns = {"Invoice":"Frequency"})

df_frequency.head()
#Her bir müşterinin daha önce hesaplamış olduğumuz TotalPrice değerlerinin toplamı, bize o müşterinin yaptığı toplam harcamayı yani "Monetary" değerini verir. İlk 5 gözlemine bakalım

df_monetary = df_satis.groupby("Customer ID").agg({"TotalPrice":"sum"}).rename(columns = {"TotalPrice":"Monetary"})

df_monetary.head()
#Concatenate işleminin düzgün yapılabilmesi için satır ve sütun sayılarında, işlemi bozacak bir uyumsuzluk olmadığı kontrolü sağlanır. Tüm RFM dfleri aynı boyuta sahiptir.

print(df_recency.shape, df_frequency.shape, df_monetary.shape)
#Tüm RFM skorlarını tek bir dataframede toplayıp ismine "df_rfm" diyelim

df_rfm = pd.concat([df_recency,df_frequency,df_monetary], axis = 1)

df_rfm.head()
#RFM Skorlarını 1 den 5 e kadar puanlayacağım.



birden_bese = ['1','2','3','4','5']

besten_bire = ['5','4','3','2','1']

#Frequency : Müşterinin toplam satın alma sayısı -> değerinin yüksek olması, puanın yüksek olması anlamına gelir.

#Bu nedenle doğru bir orantı vardır. 1 den 5 e puanladım. Her sınıfın sayılarına baktım

cut_bins = [0,1,2,4,10,210]

df_rfm["FrequencyScore"] = pd.cut(df_rfm["Frequency"], bins = cut_bins, labels = birden_bese)

df_rfm["FrequencyScore"].value_counts()
#Recency : Müşterinin son temasından bugüne kadar geçen süre -> değerin yüksek olması, puanın düşük olması anlamına gelir.

#Bu nedenle ters bir orantı vardır. 5'ten 1'e puanladım. Her sınıfın sayılarına baktım

df_rfm["RecencyScore"] = pd.qcut(df_rfm['Recency'], 5, labels = besten_bire)

df_rfm["RecencyScore"].value_counts()
#Monetary : Müşterinin yaptığı toplam harcama -> değerin yüksek olması, puanın yüksek olması anlamına gelir.

#Bu nedenle doğru bir orantı vardır. 1 den 5 e puanladım. İlk 5 gözlemi aldım. Her sınıfın sayılarına baktım

df_rfm["MonetaryScore"] = pd.qcut(df_rfm['Monetary'], 5, labels = birden_bese)

df_rfm["MonetaryScore"].value_counts()
#RFM skorunun müşteri bazında birleştirilmiş halini RFM_SCORE olarak dataframe yapımıza ekleyelim. İlk 5 gözlemi alalım.

df_rfm["RFM_SCORE"] = df_rfm['RecencyScore'].astype(str) + df_rfm['FrequencyScore'].astype(str) + df_rfm['MonetaryScore'].astype(str)

df_rfm.head()
#En Yüksek RFM_SCORE a sahip ilk 5 müşteri

df_rfm[df_rfm["RFM_SCORE"]=='555'].head()
#En Düşük RFM_SCORE a sahip ilk 5 müşteri

df_rfm[df_rfm["RFM_SCORE"]=='111'].head()
#RFM skorlarının istatistiksel değerlerine bakalım

df_rfm.describe().T

#Recency ve Frequency Skorlarına göre Müşteri Segmentlerine isim verebilmek için regular expression (regex) yapısı kuruyorum.

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
#Regex yapısını kullanarak müşterilerimi segmentlere ayırıyorum ve bulundukları segmentleri "Segmet" kolonunda gösteriyorum. İlk 5 gözleme bakalım

df_rfm['Segment'] = df_rfm['RecencyScore'].astype(str) + df_rfm['FrequencyScore'].astype(str)

df_rfm['Segment'] = df_rfm['Segment'].replace(seg_map, regex=True)

df_rfm.head()
#"Need Attention" sınıfına ait customer ID'leri seçerek excel çıktısı almak istiyorum.

need_attention_df = pd.DataFrame()

need_attention_df["NeedAttentionCustomerID"]= df_rfm[df_rfm["Segment"]=='Need Attention'].index

need_attention_df.head()
#Excele alıyorum

need_attention_df.to_csv("Need_Attention.csv")
#Her bir RFM grubunun ortalama ve adet değerlerini görmek istiyorum.



df_rfm[["Segment", "Recency","Frequency","Monetary"]].groupby("Segment").agg(["mean","count"])