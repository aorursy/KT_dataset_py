# Gerekli import işlemleri gerçekleştirildi.
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
#Tüm sutünları ve satırları gözlemlemek için kullandığımız kod.
pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);
# Veri seti okunuldu. Veri setinde sutünlarda değişkenler olmadığı için names argümanı ile değişkenlere isimler atandı.
df = pd.read_csv('../input/real-supermarket-data/satislar.csv',names = ['BranchId', 'PosId', 'InvoiceDate', 'InvoiceNo','StockCode','Line','Quantity','CategoryCode','CategoryName'],
                 low_memory=False,sep = ";", header = None)
# Veri setine ait ilk 5 gözleme erişildi.
df.head()
# Veri setinin boyutuna erişildi.
df.shape
# Veri setinde eksik gözlem var mı sorusuna yanıt alındı.
df.isnull().sum()
# CategoryCode ile birliktelik analizi yapmayacağız için silebiliriz.
df.dropna(subset= ["CategoryCode"],inplace= True)
df.shape
# Veri setinde eksik gözlem olup olmadığı kontrol edildi.
df.isnull().sum()
# Veri setinde her değişken için eşsiz gözlem sayısı kaç tane buna erişildi.
df.nunique()
#hangi üründen kaçar tane var?
df["CategoryName"].value_counts().head()
# Veri setinin betimsel istatistiklerine erişim sağlandı.
df.describe().T
# Veri setindeki CategoryName değişkeninde yer alan ürünleri strip komutu ile ayırdık.
df['CategoryName'] = df['CategoryName'].str.strip(',')
df.head()
#en çok sipariş edilen ürün hangisi?
df.groupby("CategoryName").agg({"Quantity":"sum"}).sort_values("Quantity", ascending = False).head()
# Veri setinde yer alan CategoryName değişkenine ait normalize ederek frekans değerleri getirilmiştir.
pd.DataFrame(df["CategoryName"].value_counts(normalize=True)).head(100)
#Veri setinde yer alan Quantity değişkenini integera çevirmekteyiz. 
df['Quantity'] = [x.replace(',', '.') for x in df['Quantity']]
df["Quantity"] = df["Quantity"].astype("float")
df["Quantity"] = df["Quantity"].astype("int")
#Veri setinin yapısal bilgilerine erişim sağlandı aynı zamanda Quantity değişkeninin integer olduğunu gözlemlemekteyiz.
df.info()
branch_order = (df.groupby(['InvoiceNo', 'CategoryName'])['Quantity'] .sum().unstack().reset_index().fillna(0) 
          .set_index('InvoiceNo'))
branch_order.head()
# Veri setini 1 ve 0 hale dönüştürme. One Hot Encoding işlemi yapılmıştır.
branch_encoded = branch_order.applymap(lambda x: 0 if x<=0 else 1) 
basket_branch = branch_encoded 
#Apriori en yaygın kullanılan fonk olarak da bilinmektedir. Support değeri için bir eşik değeri seçilir ve support değeri hesaplanır.
frq_items = apriori(basket_branch, min_support = 0.01, use_colnames = True)
# Support işleminin ilk beş gözlem birimine erişilmiştir.
frq_items.head()
#association_rules fonk kullanılarak "confidence ve lift" değerlerimize erişmiş olduk. Peki bu değerler neyi ifade etmektedir.
#Support: Örneğin X ve Y ürünleri olsun. Bu ikisinin birlikte görülme olasılığını ifade etmektedir.
#Confidence: Örneğin X ve Y ürünleri olsun. X'i alanların % şu kadarı Y'yide alacaklar. Bu bize confidence değerini verecektir.
#Lift: Bir ürünün alınması diğer ürünün alınmasını % kaç arttırmaktadır. Bizlere bu bilgiyi vermektedir. 
#Bu bizim için aynı zamanda final tablosudur.
rules = association_rules(frq_items, metric ="confidence", min_threshold = 0.20) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
rules.head(50) 