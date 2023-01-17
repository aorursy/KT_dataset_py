import pandas as pd
import numpy as np
import seaborn as sns

# to display all columns and rows:
pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);

#virgulden sonra gösterilecek olan sayı sayısı
pd.set_option('display.float_format', lambda x: '%.0f' % x)
import matplotlib.pyplot as plt
#Veriyi okuma
df_2010_2011 = pd.read_excel("../input/uci-online-retail-ii-data-set/online_retail_II.xlsx", sheet_name = "Year 2010-2011")
#Verinin kopyasını alma
df = df_2010_2011.copy()
df.head()
#essiz urun sayisi, kac musteri var, vs...
df.nunique()
#hangi urunden kacar tane var?
df["Description"].value_counts().head()
#en cok siparis edilen urun hangisi, sıralama ile birlite?
df.groupby("Description").agg({"Quantity":"sum"}).sort_values("Quantity", ascending = False).head()
#TotalPrice değişkeni oluşturduk.Toplam harcamalar için.
df["TotalPrice"] = df["Quantity"]*df["Price"]
df.groupby("Invoice").agg({"TotalPrice":"sum"}).head()
#hangi ulkeden kac siparis geldi?
df["Country"].value_counts().head()
#hangi ulke ne kadar kazandırdı?
df.groupby("Country").agg({"TotalPrice":"sum"}).sort_values("TotalPrice", ascending = False).head()
#Kaç tane eksik değerimiz var?
df.isnull().sum()
#Yukarıdaki çıktıya göre StockCode u belli olan 1454 ürünün Description ı eksik.
#Bunları düzeltebilir miyiz diye merak ediyorum ve bulmaya çalışıyorum.
#Öncelikle eksik olan değerlerin Stockcode unu bulmam gerekiyor.
df_eksik_desc = df[df["Description"].isnull()][["StockCode"]]
df_eksik_desc.head()
df_eksik_desc.size
#Şimdi bulduğumuz bu StockCode ların kaçı eşsiz ona bakalım.
df_eksik_desc.nunique()
#Burda tekrarlayan kayıtları siliyorum.
df_eksik_desc.drop_duplicates(inplace=True) 
df_eksik_desc.size
df_eksik_desc.head()
#Yukarıda bulduğumuz sonuç Description ı eksik StockCode u belli 633 değer olduğunu gösteriyor.
#Bu değerlerden birisini ele alalım.
df[df["StockCode"]==22139].head(20)
#Yukarıdaki tablodan çıkardığım  bazı sonuçlar.
#1-Descriptionu bir gözlemde olupta başka bir gözlemde olmayan değerler var.Burada eksik değerler doldurulabilir.
#2-StockCode u aynı olup, descriptionu tamamen farklı olan bir değer var.(3. satırdaki amazon değeri)
#3-Aynı tarihlerde farklı fiyata satılan aynı ürün.İndirim ve ya kampanya olarak değerlendiremeyiz.Aynı ülkelere.
#4-Bazı gözlemlerde price 0.İadelerde bile price yazıyor.
#5-Invoice u C ile başlamayan buna rağmen eksi değerde Quantity var.
#Burada çoğunluğa göre hareket edip eksik olan değerleri Descriptiona göre dolduracağız. 
#Uzun süreli bir işlem.
#RFM için gereksiz ama ürünler bazında yapılması gereken bir işlem
dongu_uzunlugu = len(df_eksik_desc["StockCode"])
sayac = 0
for i in df_eksik_desc["StockCode"]:
    if(len(df[df["StockCode"]==i]["Description"].value_counts())>0): #Eğer StockCode u ile Descriptionu eşleşiyorsa
        a = df[df["StockCode"]==i]["Description"].value_counts().index[0]  #En çok olan değeri alıyoruz.
        df.loc[(df["StockCode"]==i) & (df["Description"]!=a),"Description"] = a  #eksik verilere atıyoruz.
    print(str(dongu_uzunlugu - sayac - 1 )+" deger kaldi")
    sayac += 1   
#Baslama saati 11.52
#Yarısı        13.22
#Bitis saati   15.04
#Geriye bulamayacağımız kaç eksik değer oldugunu görüyoruz.112 değer bulunamayacak şekilde gözüküyor
#Geriye bulamayacağımız kaç eksik değer oldugunu görüyoruz.112 değeri en son silicez.
df.isnull().sum()
#Şimdi ise CustomerId leri eksik olan Invoice lerini (Fatura no) bulup buna göre belirlemeye çalışagız.
df_eksik_cust = df[df["Customer ID"].isnull()][["Invoice"]]
df_eksik_cust.head()
df_eksik_cust.size
df_eksik_cust.nunique()
#Burda tekrarlayan kayıtları siliyorum.
df_eksik_cust.drop_duplicates(inplace=True) 
df_eksik_cust.size
df_eksik_cust.head()
#Uzun süreli
#Burada Customer ID si boş olan Invoice lerde yapısal bozukluk olabilir mi diye kontrol ettim.
toplam = 0
for i in df_eksik_cust["Invoice"]:
    if(len(df[df["Invoice"]==i]["Customer ID"].value_counts())>0): #Eğer Invoice u ile Customer ID ile eşleşiyorsa
        toplam += 1 
print(toplam)
df.shape
#Şimdi eksik olan değerleri Customer ID özelinde siliyoruz.
#Description eksik olması RFM için bir anlam ifade etmiyor.
df.dropna(subset= ["Customer ID"],inplace= True)
#Zaten Description daki eksik değerler de bunun içinde.
df.isnull().sum()
df.shape
#Invoice değeri C ile başlayan iade anlamına gelen değerleri siliyoruz.
sil =  df[df["Invoice"].str.contains("C", na=False)].index
df = df.drop(sil, axis=0)
df.shape
#Son olarak iade olmadığı halde Quantity değeri eksi olan değerleri siliyoruz.
sil_q = df[~df['Quantity'] > 0].index
df = df.drop(sil_q, axis=0)
#Burada değişim olmamasının nedeni bu değerlerin demek ki eksik değerlerle birlikte silinmesidir.
df.shape
df.describe([0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95, 0.99]).T
#Kaç aykırı değer olduğunu gösteriyor.
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
#Verileri düzenledikten sonra oluşan son tablonun özet bilgileri
df.info()
#Yapılan ilk alışveriş tarihi
df["InvoiceDate"].min()
#Yapılan son alışveriş tarihi
df["InvoiceDate"].max()
#Recency değeri belirlemek için bugunun tarihini alıyoruz.
import datetime as dt
today_date = dt.datetime(2011,12,10)
df.groupby("Customer ID").agg({"InvoiceDate":"max"}).head()
#Su an her bir müşterinin son alışveriş tarihleri elimizde.
df["Customer ID"] = df["Customer ID"].astype(int)
#Recency değeri için bugunun tarihi ile verimizdeki tarihleri çıkardık.
temp_df = (today_date - df.groupby("Customer ID").agg({"InvoiceDate":"max"}))
temp_df.rename(columns={"InvoiceDate": "Recency"}, inplace = True)
temp_df.head()
#Sadece günler ile işlem yapacağımız için sadece günleri tutuyoruz.
recency_df = temp_df["Recency"].apply(lambda x: x.days)
recency_df.head()
#Bir kullanıcının her bir faturasının içinde kaç ürün olduğu
temp_df = df.groupby(["Customer ID","Invoice"]).agg({"Invoice":"count"})
temp_df.head()
#Burda ise kullanıcının kaç faturası olduğu
temp_df.groupby("Customer ID").agg({"Invoice":"count"}).head()
freq_df = temp_df.groupby("Customer ID").agg({"Invoice":"count"})
freq_df.rename(columns={"Invoice": "Frequency"}, inplace = True)
freq_df.head()
#Kişinin ne kadar harcama yaptığı
monetary_df = df.groupby("Customer ID").agg({"TotalPrice":"sum"})
monetary_df.head()
monetary_df.rename(columns={"TotalPrice": "Monetary"}, inplace = True)
print(recency_df.shape,freq_df.shape,monetary_df.shape)
rfm = pd.concat([recency_df, freq_df, monetary_df],  axis=1)
rfm.head()
rfm["RecencyScore"] = pd.qcut(rfm['Recency'], 5, labels = [5, 4, 3, 2, 1])
rfm["FrequencyScore"] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels = [1, 2, 3, 4, 5])
rfm["MonetaryScore"] = pd.qcut(rfm['Monetary'], 5, labels = [1, 2, 3, 4, 5])
rfm.head()
rfm["RFM_SCORE"] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str) + rfm['MonetaryScore'].astype(str)
rfm.head()
rfm.describe().T
#Sınıflandırmaya yarayan regex değişkeni
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
rfm['Segment'] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str)
rfm['Segment'] = rfm['Segment'].replace(seg_map, regex=True)
rfm.head()
rfm[["Segment", "Recency","Frequency","Monetary"]].groupby("Segment").agg(["mean","count"])
rfm[rfm["Segment"] == "Need Attention"].head()
rfm[rfm["Segment"] == "Need Attention"].index
new_df = pd.DataFrame()
new_df["Need Attention ID"] = rfm[rfm["Segment"] == "Need Attention"].index
new_df.head(10)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", context="talk")

# Set up the matplotlib figure
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 10), sharex=True)

#Recency
x = rfm["Segment"].values
y1 = rfm["Recency"]
sns.barplot(x=x, y=y1, palette="rocket", ax=ax1)
ax1.axhline(0, color="k", clip_on=False)
ax1.set_ylabel("Recency")

#Frequency
y2 = rfm["Frequency"]
sns.barplot(x=x, y=y2, palette="rocket", ax=ax2)
ax2.axhline(0, color="k", clip_on=False)
ax2.set_ylabel("Frequency")

#Monetary
y3 = rfm["Monetary"]
sns.barplot(x=x, y=y3, palette="rocket", ax=ax3)
ax3.axhline(0, color="k", clip_on=False)
ax3.set_ylabel("Monetary")

# Finalize the plot
sns.despine(bottom=True)
plt.setp(f.axes, yticks=[])
plt.tight_layout(h_pad=2)
# Utku Özkan tarafından alıntı. https://www.kaggle.com/utkuzkan/rfm-customer-segmentation-analysis
# Segmentler içinde kaç kişi var.
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
        if segments_counts.index[i] in ['Champions', 'Loyal Customers']:
            bar.set_color('firebrick')
        ax.text(value,
                bar.get_y() + bar.get_height()/2,
                '{:,} ({:}%)'.format(int(value),
                                   int(value*100/segments_counts.sum())),
                va='center',
                ha='left',
                fontsize=14,
                  weight='bold'
               )

plt.show()
#Son olarak iadeleri en çok hangi segment gerçekleştirmiş.
df_iade = df_2010_2011[df_2010_2011["Invoice"].str.contains("C", na=False)]
df_iade.head()
df_iade.shape
#İadeleri fatura bazında tutmak için tekrarlayan kayıtları siliyoruz.
df_iade.drop_duplicates(subset ="Invoice",  keep = "first", inplace = True) 
df_iade.shape
df_iade.rename(columns={"Invoice":"Iade Sayisi"}, inplace= True)
df_iade.groupby("Customer ID").agg({"Iade Sayisi": "count"}).sort_values("Iade Sayisi",ascending=False).head()
df_iade_rfm = rfm.merge(df_iade,on="Customer ID")
df_iade_rfm = df_iade_rfm.groupby("Segment").agg({"Iade Sayisi":"count"})
df_iade_rfm
# Set up the matplotlib figure
f, ax1 = plt.subplots(1, 1, figsize=(20, 10), sharex=True)

#İade
x = df_iade_rfm.index
y1 = df_iade_rfm["Iade Sayisi"]
sns.barplot(x=x, y=y1, palette="rocket", ax=ax1)
ax1.axhline(0, color="k", clip_on=False)


# Finalize the plot
sns.despine(bottom=True)
plt.setp(f.axes)
plt.tight_layout(h_pad=2)
