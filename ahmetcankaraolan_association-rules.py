#Gerekli import işlemleri gerçekleştirilmiştir. 

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
#Veri seti okunmuştur.Veri setinde kolon (değişken) bulunmamaktaydı. Buraya names argümanı ile bir isim ekleyebiliriz. Products olarak. 
df = pd.read_csv("../input/association-rules-analysis/datasets_344_727_GroceryStoreDataSet.csv", names = ["products"], header = None)
#Veri Setine ait ilk 5 değer gözlemlenmiştir.
df.head()
#Dataframe'in boyutu incelendi.
df.shape
#Birliktelik analizi için gerekli olan kütüphane kurulumu gerçekleştirilmiştir.
!pip install mlxtend
#df içerisinden products değişkeni seçilip burada products değişkeninde yer alan ürünler bir arada iken isimsiz fonskiyon ile birbirlerinden ayrılmıştır. 
items = list(df["products"].apply(lambda x:x.split(',')))
items 
#Birliktelik analizinde dataframe ya 1 ve 0'lardan ya da True False şeklinde Boolen veri yapısı tipinde olmak zorundadır. Bunu yapabilmek için mlxtend içinden TransactionEncoder fonk. aktif edilmiştir.
from mlxtend.preprocessing import TransactionEncoder
#TransactionEncoder fonk fit_transfor edilerek verilerimiz artık True ve False olarak dönüştürülmüştür. 
x = TransactionEncoder()
z = x.fit_transform(items)
x.columns_
#Dönüştürülen veri seti datafarme olarak kaydedildi ve değişken adına ürünlerin isimleri verilmiştir. 
df = pd.DataFrame(z,columns=x.columns_)
#Dataframe artık birliktelik analizi için uygun hale gelmiştir. Bir sonraki adım olan support ve confidence değerlerini hesaplayıp yorumlamak olacaktır.
df
#Apriori en yaygın kullanılan fonk olarak da bilinmektedir. Support değeri için bir eşik değeri seçilir ve support değeri hesaplanır.
freq_items = apriori(df, min_support = 0.2, use_colnames = True, verbose = 1)
#Ürünlerimizin support değerine ulaştık. Support 
freq_items.head()
#association_rules fonk kullanılarak "confidence ve lift" değerlerimize erişmiş olduk. Peki bu değerler neyi ifade etmektedir.
#Support: Örneğin X ve Y ürünleim olsun. Bu ikisinin birlikte görülme olasılığını ifade etmektedir.
#Confidence: Örneğin X ve Y ürünleim olsun. X'i alanların % şu kadarı Y'yide alacaklar. Bu bize confidence değerini verecektir.
#Lift: Bir ürünün alınması diğer ürünün alınmasını % kaç arttırmaktadır. Bizlere bu bilgiyi vermektedir. 
#Bu bizim için aynı zamanda final tablosudur.

association_rules(freq_items, metric = "confidence", min_threshold = 0.5)