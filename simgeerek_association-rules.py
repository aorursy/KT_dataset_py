# gerekli kütüphaneler import edildi, tüm satır ver sutunları görmek için ayarlar yapıldı.

import pandas as pd

import numpy as np

from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
# veri seti okutuldu, ilk beş gözleme bakıldı.

df = pd.read_csv("../input/grocerystoredataset/GroceryStoreDataSet.csv", names =['Products'])

df.head()
# Veri setinin boyutu

df.shape
# product sutunundaki veriler listeye atandı

data = list(df["Products"].apply(lambda x:x.split(',')))

data 
# listeyi True ve false hale dönüştürme. One Hot Encoding işlemi yapılmıştır.

from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()

te_data = te.fit(data).transform(data)

df = pd.DataFrame(te_data,columns=te.columns_)

df.head()
# Apriori en yaygın kullanılan fonk olarak da bilinmektedir. Support değeri için bir eşik değeri seçilir ve support değeri hesaplanır.

freq_items = apriori(df, min_support = 0.2, use_colnames = True, verbose = 1) 
# bulunan support değerlerine erişilir.

freq_items
#association_rules fonk kullanılarak "confidence ve lift" değerlerimize erişiyoruz

rules = association_rules(freq_items, metric="confidence", min_threshold = 0.6)

rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 

rules 