# Kullanacağımız modül ve kütüphaneler
import pandas as pd
import numpy as np

# Verilerimizi ilgili veri setinden okuyoruz
nlf_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")
sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")

# Tekrar işlemleri için (np.random.seed(0)) kodunu ekliyoruz.
np.random.seed(0) 
# Veri setimizin (nlf_data) ilk birkaç satırına bakarak eksik değerleri kontrol edebiliriz.
nlf_data.sample(5)
# sf_Permits veri setimiz içeriğini 5 satırlık bir özetine bakarak eksik veri olup olmadığını gözlemleyeceğiz.
sf_permits.sample(5)
#Şimdi her kolonda kaç tane eksik değer var onlara bakalım
missing_values_count = nlf_data.isnull().sum()

#Şimdi de  ilk 10 sütundaki eksik değerlerin kaç tane olduğuna bakalım
missing_values_count[0:10]
#Toplam kaç tane eksik verimiz var ona bakalım
total_cells = np.product(nlf_data.shape)
total_missing = missing_values_count.sum()
#Eksik verilerin yüzde(%) olarak ne kadar olduğuna bakalım
(total_missing/total_cells) * 100
#Peki sf_permits veri setimizdeki eksik verilerin oranını merak etmiyor musunuz ? Hadi birde ona bakalım
total_cells = np.product(sf_permits.shape)
total_missing = missing_values_count.sum()
#Şimdi de % oranını görüntüleyelim
(total_cells / total_missing)*100
#şimdi ilk 10 satırdaki eksik değerlere bakalım
missing_values_count[0:10]
#Evet şimdi tüm satırlardaki eksik(NaN) değerlerini siliyoruz
nlf_data.dropna()
#Bir tane bile eksik değer olan tüm kolonları silelim
colomns_with_na_dropped = nlf_data.dropna(axis=1)
colomns_with_na_dropped.head()
#Şimdi bakalım ne kadar değer kayıbımız var :)
print("Orijinal veri setinin sütunları : %d \n" % nlf_data.shape[1])
print("Eksik verileri çıkarıldıktan sonraki sütun sayısı : %d" % colomns_with_na_dropped.shape[1])
#sf_permits veri setinden bir tane bile eksik değer olan tüm kolonları silelim
colomns_with_na_dropped = sf_permits.dropna(axis=1)
colomns_with_na_dropped.head()
#Gelin birde sf_permits veri setimizdeki eksik değer olan sütunları silelim, ne dersiniz ?
print("Orginal veri setinin sutunları : %d \n" % sf_permits.shape[1])
print("Eksik değerler silindikten sonra sf_permits veri setindeki  sütün sayısı : %d" % colomns_with_na_dropped.shape[1])
#NFL veri setinden bir alt küme veri seti alıyoruz
subset_nlf_data = nlf_data.loc[:, 'EPA':'Season'].head()
subset_nlf_data
#Tablodaki NaN değerlerinin yerine hepsine sıfır'0' yazıyoruz.
subset_nlf_data.fillna(0)
#Aynı sütunlardaki tüm NaN değerlerini  sıfır(0) ile değiştiriyoruz 
subset_nlf_data.fillna(method = 'bfill', axis = 0).fillna("0")
# Pratik ! sf_permits veri setindeki NaN değerlerini sıfır '0' ile değiştiriniz.
# Sonra Aynı sütunda bulunan NaN değerlerini sıfır ile değiştiriniz.