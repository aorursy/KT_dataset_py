from mpl_toolkits.mplot3d import Axes3D  #Gerekli kütüphanelerimizi import ediyoruz. Seaborn grafikler için sklearn öğrenme adımları için decision tree ve knn için vs...
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score, recall_score
from sklearn import utils
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
print(os.listdir('../input'))
nRowsRead = 100
dataway = pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv', delimiter=',' ,nrows = nRowsRead) #Data setimizi pandas ile okumamız için yolu belirtiyoruz.
dataway.dataframeName = 'Video_Games_Sales_as_at_22_Dec_2016.csv' #Shape ve print kullanarak row ve columnları yazdırıyoruz.
[nRow, nCol] = dataway.shape
print(f'Bakılan {nRow} satır ve {nCol} sütun')
dataway.head(7) #Baştan başlayarak datasetimizden aldığımız öznitelik tablosunun 7 elemanını görüntülüyoruz.
dataway.info() #İnfo komutu ile değerlerin doluluk veya eksiklik durumunu görüntülüyoruz.
dataway.tail() #Sondan 5 elemanı görüntülüyoruz.
plt.hist(dataway['Global_Sales']) #Pandas ile kullanmak için oluşturduğumuz veri yolunun yani datasetimizin 'Global_Sales' özniteliğinin histogramını çizdiriyoruz.
plt.title('Global Sales Histogram') #Histogram adı.
plt.show 
plt.hist(dataway['NA_Sales']) #Datasetimizin 'NA_Sales' özniteliğinin histogramını çizdiriyoruz.
plt.title('North America Sales') #Histogram adı.
plt.show
corrmat = dataway.corr(method='spearman') #Bu kısımda kolerasyon tablosu çizdiriyoruz.
f, ax = plt.subplots(figsize=(12, 10))    # Düz metin ve ısı haritaları şekildeki gibi gözükmekte.
sns.heatmap(corrmat, ax=ax, cmap="YlGnBu", linewidths=0.1)
corrmat = dataway.corr(method='spearman') # Yine aynı şekilde ayrıntılı kolerasyon çizimi yapıyoruz.
cg = sns.clustermap(corrmat, cmap="YlGnBu", linewidths=0.1);
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
cg
sns.jointplot(x='NA_Sales', y='Global_Sales', data=dataway[dataway['NA_Sales'] < 100], kind='hex',gridsize=20) #NA_Sales özniteliğinin 100 den küçük olanlarının Global_Sales e göre ısı haritası.
              
missing_values_count = dataway.isnull().sum() #Eksik değerlerin bulunduğu öznitelikleri buluyoruz.
missing_values_count[0:17] #0-17 ye kadar olan kolonları gösterdim.
dataway.dropna() #Eksik değer olan rowları kaldırdık! 36 Adet bulduğumuz eksik değer 100 adet rowdan kaldırılınca 64 adet rowumuz kaldı.Tekrardan eklicez...
columns_with_na_dropped = dataway.dropna(axis=1) #Düşen kolonları/kaldırılanları yazdırmak için değişkene atıyorum
columns_with_na_dropped.head() # 
print("Datasetteki orijinal sütun : %d \n" % dataway.shape[1]) #Orijinal sütun ve droplanmış sütunları yazdırıyoruz.
print("na's dropped sütun: %d" % columns_with_na_dropped.shape[1])
subset_dataway = dataway.loc[:, 'Name':'Rating'].head(7) #Rating e kadar olan boşları NaN ile dolduruyoruz.
subset_dataway
subset_dataway.fillna(0) #Boşları(NaN) 0 la dolduruyoruz.
subset_dataway.fillna(method = 'bfill', axis=0).fillna(0) #0 lanan öznitelikleri alttaki kolondaki veri ile dolduruyoruz.
dataway.max #Max değerlerinin bulunması
dataway.min #Min değerlerinin bulunması
year_release = pd.to_datetime(dataway['Year_of_Release']) #Yayınlanma zamanındaki yıl bilgisini kullanarak 'New_or_Old' isimli yeni bir öznitelik oluşturduk.
dataway['New_or_Old'] = year_release.dt.year #Seçtiğimiz dataset pek uygun olmasada teorik olarak bu şekilde öznitelik çıkarımında bulunabiliriz.
dataway.head() #Uzun olmasın diye ilk 5 te yeni özniteliğimizi görebiliriz.
x = dataway[['User_Score']].values.astype('float64')#User_Score özniteliğini normalleştirmek için seçiyoruz.
min_max_scaler = preprocessing.MinMaxScaler() #Normalleştirmek için MinMax normalleştirme metodunu kullanıyoruz.
x_scaled = min_max_scaler.fit_transform(x)
dataway['User_Score_2'] = pd.DataFrame(x_scaled)
dataway.head()
selected_features = ['NA_Sales','Global_Sales','EU_Sales','JP_Sales']
defining_columns = dataway[selected_features]

defining_columns #Gösterme
prediction_column = dataway.Other_Sales #Tahmin etmek istediğimiz özniteliği prediction_column değişkenine atıyoruz
prediction_column.describe() #Gösterimini yapıyoruz hata olmasın diye.
defining_columns_train,defining_columns_test,prediction_column_train,prediction_column_test = train_test_split(defining_columns,prediction_column,test_size = 0.2, random_state = 0)
#test verilerini bölüyoruz %20 test %80 eğitim için ayırdık
regressor = DecisionTreeRegressor(random_state = 0) #Decision tree regresyon gerçekleştirimi.
regressor.fit(defining_columns_train,prediction_column_train)
prediction_column_pred = regressor.predict(defining_columns_test)
other_sales_prediction = regressor.predict(defining_columns_test) #Tahmin ettiğimiz değerlerle gerçek değer ne kadar yakın ?
mean_absolute_error(prediction_column_test,other_sales_prediction) #Tahmin ne kadar yakınsa fark 0'a o kadar yakın olur.
selected_features = ['NA_Sales','Global_Sales','EU_Sales','JP_Sales']
defining_columns = dataway[selected_features]

prediction_column = dataway.Other_Sales
prediction_column.describe()
forest_model = RandomForestRegressor(random_state = 0) #Random Forest Regresyon Gerçekleşimi
forest_model.fit(defining_columns_train,prediction_column_train)
prediction_column_pred = forest_model.predict(defining_columns_test)
other_sales_prediction = forest_model.predict(defining_columns_test)#Tahmin ettiğimiz değerlerle gerçek değer ne kadar yakın ?
mean_absolute_error(prediction_column_test,other_sales_prediction)#Tahmin ne kadar yakınsa fark 0'a o kadar yakın olur
#Gerçek değer ile tahmini değer arasındaki farkın ortalaması Decision Tree'de => '0.5249999999999999' 
#Gerçek değer ile tahmini değer arasındaki farkın ortalaması Random Forest'de => '0.4226' 
#Bu verilere göre farkın ortalaması az olan random forest regresyonu daha iyi sonuç verdiğinden dolayı bizim problemimiz için daha kulanılabilir.
#Eğer ikiside kötü çıksaydı; Modeli eğitmekte kullandığımız kolonlardan daha uygunları seçilirse (feature selection, extraction) tahmin yeteneği gelişebilir.
#Baska algoritmalar kullanıllabilir/test edilebilir.
#Varsa modelin parametrelerini ayarlamak performansını arttırabilir.