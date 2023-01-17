#https://www.kaggle.com/shazadudwadia/supermarket linkindeki dataseti "Add Data" butonu ile notebookuma dahil ettim. Ardından CSV okuma işlemini gerçekleştiriyorum.



import pandas as pd

df = pd.read_csv('../input/supermarket/GroceryStoreDataSet.csv', names=['products'], header=None)

df
#DataFrame içerindeki verileri merak ettim, verileri çekiyorum.



df.values
#Satır ve Sütun bilgisini merak ettim, boyut bilgisine erişiyorum.



df.shape
#Adım1 : Veriyi liste formatına dönüştürdüm. Her bir satırdaki objeleri ',' ile ayırdım.



data = list(df["products"].apply(lambda x : x.split(',')))

data
#Adım2 : mlxtend kütüphanesinin veriyi True-False dataframe ine dönüştürme metodunu uyguluyacağım. 

#İlk önce, kurulu olmayanlar için mlxtend kütüphanesini install ediyorum.



!pip install mlxtend



from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()

te_data = te.fit(data).transform(data)

df = pd.DataFrame(te_data,columns=te.columns_)

df
# Apriori algoritmasına min_support değeri 0.2 verildi. Böylece kombinasyonlardaki 0.2 support değerinin altında olan ürün birliktelikleri elenmiş oldu.

# "verbose" argümanının 1 olması ise toplam kaç kombinasyon olduğu bilgisini bize verecektir. Örneğimizde 42 kombinasyon oluşmuştur. 

# En son durumda ise elimizde 16 kombinasyon kalmıştır. 

# Demek ki 42-16 = 26 kombinasyonumuz 0.2 support değerinin altında kalmış ve yorumlarımıza katmayacağımız önemsiz bir oran olarak ele alınmıştır.



from mlxtend.frequent_patterns import apriori

freq_items = apriori(df,min_support=0.20,use_colnames = True, verbose = 1)

freq_items
# "min_threshold = 0.3" verilerek, "confidence" değeri 0.3 altında olan değerlerin getirilmemesi tercih olarak sağlandı.  



from mlxtend.frequent_patterns import association_rules

df_res = association_rules(freq_items, metric = "confidence", min_threshold = 0.3)

df_res
#En yüksek confidence değerini bulalım. Çıktıda en yüksek confidence değerinin 0.80 olduğu görülüyor.



conf_max = df_res['confidence'].max()

conf_max
#En düşük confidence değerini bulalım. Çıktıda en düşük confidence değerinin 0.307 olduğu görülüyor.



conf_min = df_res["confidence"].min()

conf_min
#En düşük, En yüksek ve 0.5 confidence değerine sahip olan veriler filtreleniyor. "confidence" değerine göre artan sırada sıralanıyor.



df_filt = df_res[ (df_res["confidence"] == conf_min) | (df_res["confidence"] == conf_max) | (df_res["confidence"] == 0.5 )]

df_filt.sort_values("confidence", ascending = True)