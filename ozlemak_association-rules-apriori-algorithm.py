import pandas as pd

import numpy as np



from mlxtend.frequent_patterns import apriori, association_rules

from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv('../input/supermarket/GroceryStoreDataSet.csv' , names =["Products"], header=None)

df.head() #Veri setinin ilk 5 gözlemi
#Veri setinin boyutları

df.shape 
#Veri setinin boyut sayısı

df.ndim
#unique değişkenlerin bulunup items olarak atanması

items = df["Products"].unique()

items
#df içerisinden products değişkeni seçilip burada products değişkeninde yer alan ürünler bir arada iken birbirlerinden ayrılarak listelenmiştir.

items = list(df['Products'].apply(lambda x : x.split(',')))

items
#müşterinin sepetinde hangi ürünlerin olup olmadığı ONE-HOT ENCODING ile boolean yapılır



te = TransactionEncoder()

te_items = te.fit(items).transform(items)

items_df = pd.DataFrame(te_items,columns=te.columns_)

items_df
#Esik değerini support'a göre belirlemek için support hesabı yapılır. Çıktıya göre 83 farklı kombinasyon ortaya çıktı. (11 tane ürünümüz vardı.) 

#Yorum örneği : Ekmek bütün alışverişlerin % 65'inde  bulunmaktadır.

freq_items  = apriori(items_df,min_support=0.01,use_colnames=True)

freq_items
final_tableau = association_rules(freq_items,metric="confidence",min_threshold = 0.6)

final_tableau
final_tableau[(final_tableau['confidence']>0.6) & (final_tableau['support']>0.3)]
final_tableau[(final_tableau['confidence']>0.6) & (final_tableau['support']>=0.2)]
final_tableau
final_tableau.sort_values(by="support",ascending=False).head(6)