import pandas as pd
pd.set_option('display.max_columns', 500)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/otomobil.csv'
                     , names=['symboling','normalized-losses','make','fuel-type'
                              ,'aspiration','num-of-doors','body-style','drive-wheels'
                              ,'engine-location','wheel-base','length','width','height'
                              ,'curb-weight','engine-type','num-of-cylinders','engine-size'
                              ,'fuel-system','bore','stroke','compression-ratio','horsepower'
                              ,'peak-rpm','city-mpg','highway-mpg','price'])
df.head()
df.tail()
df.info() #Veriler hakkında genel bir bilgi [veri tipi, eksik değer sayısı, satır sayısı]
df.shape #rows and columns of data set / Data setimizin satır ve sütun sayısı
df.describe(include='all')
x=df
y=df.iloc[:,0]  
#sembol sütununu y değişkenine atıyoruz.
y[:].value_counts()
#Sembollerin temsil ettikleri araba sayısı.
y[:].astype(int).plot.hist();
##Sembollerin temsil ettikleri araba sayısı histogram gösterimi. [0 ve 1 yoğunlukta]
x=x.replace('?', np.nan)
#Boş hücrelerin değerini global bir değerle dolduruyoruz.[NaN]
a=df[['symboling','height','wheel-base','length','price']]
sns.pairplot(a, hue='symboling', size=2.5);
#Özniteliklerin semboller cinsinden yoğunluklarının pairplot gösterimi.
data_corr=df.corr() #Korelasyon değerlerini data_corr değişkenine atadık.
data_corr #korelasyonları görüyoruz. semboling ve height değerleri kısmende olsa korele diyebiliriz.
#Daha güçlü korelasyon değerleri var ancak bizim ilgilendiğimiz kısım symboling.
#daha ayrıntılı görelim.
data_corr['symboling'].sort_values(ascending=False)
#Semboling sütunu ile diğer özniteliklerin korelasyonlarını görüyoruz. 
#En çok korele olanlar wheel-base ve height öznitelikleri bizim işimize yarayacak
df['body-style'].value_counts().plot(kind='bar',color='r')
plt.style.use('dark_background')
plt.title("Car Type Density Chart")
plt.ylabel('Number of Vehicles')
plt.style.use('dark_background')
plt.xlabel('Car Types')
#Araba tipi yoğunluklarının gösterimini yaptık. Sedan ve hatchback yoğunlukta.
plt.figure(figsize=(10, 5))
plt.style.use('dark_background')
sns.countplot(x='make', data=df)
plt.xticks(rotation='vertical')
plt.title('Manufacturer')
plt.show()
#En çok hangi arabalar satılıyor? sorusunun yanıtı olan gösterim. Toyota diğer markalara göre açıkça daha çok tercih ediliyor
#Kayıp değerlerin sütunun yüzde kaçını oluşturduğunu bulmak için tablo oluşturduk.
def missing_values_table(df):
        # Toplam kayıp veri
        mis_val = df.isnull().sum()
        
        # Eksik değerlerin yüzdesini hesapladık.
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Bu verileri bir tabloda göstermek istiyoruz.
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Tablonun kolon isimlerini düzenleyelim.
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Büyükten küçüğe yüzdeleri sıralayarak yazdırmak istiyoruz.
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Kayıp verilerle ilgili bazı verileri not olarak yazdırıyoruz.[kaç kolonda kaç kayıp veri var?]
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Tabloyu döndürüyoruz.
        return mis_val_table_ren_columns
missing_values = missing_values_table(x)
missing_values.head()#ilk 5 değeri görüyoruz tabloda.
Q1 = x.quantile(0.25)
Q3 = x.quantile(0.75)
IQR = Q3-Q1
print(1.5*IQR)
#Çeyrekler üzerinden IQR yi buduk. 
print("Upper Limit")
print("------------------------")
print(Q1-IQR)
print("------------------------")
print("Lower Limit")
print("------------------------")
print(Q3+IQR)
print("------------------------")
#Upper ve Lower değerleri arasında değilse veriyi uç değer olarak kabul ediyoruz.
x = x[(x.price >= 0 ) & (x.price <=25212.0)]
x.shape
#Uç olan price değerlerini verilerimizden çıkarttık.
#205 ten 188 satıra düştü.
#Verilerimizin tip dönüşümünü yapıyoruz.
x['normalized-losses']=x['normalized-losses'].astype(str).astype(float)
x['bore']=x['bore'].astype(str).astype(float)
x['stroke']=x['stroke'].astype(str).astype(float)
x['price']=x['price'].astype(str).astype(float)
x['horsepower']=x['horsepower'].astype(str).astype(float)
x['peak-rpm']=x['peak-rpm'].astype(str).astype(float)
#boş değerleri ortalama değerlerle dolduruyoruz.
x=x.fillna(x.mean())
#Kapı sayısı belli olmayan araçları en çok tekrar eden değerle dolduruyor.[4 kapılı olarak varsaydık.]
x = x.fillna(df['num-of-doors'].value_counts().index[0])
missing_values = missing_values_table(x)
missing_values.head()
#Boş değer kontrolünü tekrar yapıyoruz metodumuzu kullanarak.
x.dtypes.value_counts()
#data type sayılarını gördük.
make_symboling_table = pd.crosstab(index=x["make"], 
                          columns=x["symboling"])


make_symboling_table .plot(kind="bar", 
                 figsize=(8,8),
                 stacked=True)
#Burada markaların hangi sembol sınıflarına dahil olduklarını[fiyat sınıfı] daha net görüyoruz.
#Sembol tahmini yapacağımız için hangi arabaların yüksek başarı ile sınıflandırılacağını kestirebiliriz.
#Bu bakımdan jaguar, porche, peugeot, volvo, saab,alfa-romero iyi bir seçenek olacaktır.
make_symboling_table = pd.crosstab(index=x["drive-wheels"], 
                          columns=x["symboling"])
make_symboling_table .plot(kind="bar", 
                 figsize=(8,8),
                 stacked=True)

#Arabaların çekiş tipine göre hangi sınıfa dahil oldukarını görüyoruz. 4çeker arabaların azlığı dikkat çekiyor.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le_count = 0

# Sütunların üzerinde geziyoruz.
for col in x:
    if x[col].dtype == 'object':
        # 2 veya daha az benzersiz kategori varsa
        if len(list(x[col].unique())) <= 2:
            # Eğitim verilerinin eğitilmesi
            le.fit(x[col])
            # Hem eğitim hem de test verilerini dönüştürüyoruz.
            x[col] = le.transform(x[col])
            # Kaç sütunda dönüşüm yapıldığını tutuyoruz.
            le_count += 1
            
print('%d columns were label encoded.' % le_count)

x = pd.get_dummies(x)
#Dönüştürülen hücreleri 0-1(dummies) değerlerle dolduruyoruz.
print('Training Features shape: ', x.shape)
#sütun sayımız 26 dan 72ye çıktı dönüşüm işleminden sonra.
x.columns
y=x.iloc[:,0] 
#Verilerimizi işleme hazır hale getirdikten sonra train ve test olarak ayırıyoruz.
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski')  
#Dikkate alınacak komşu sayısını 5 ve uzaklık metriği olarak minkowski'yi kullandık.
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)  
#x_test değerlerinin sınıflarını tahmin ettik.
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
cm = confusion_matrix(y_test, y_pred)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['-2','-1', '0','1','2','3']); ax.yaxis.set_ticklabels(['-2','-1', '0','1','2','3']);

#Sınıflandırma başarısı düşük çıktı. Bir çok sebeple olabilir.
#İlk akla gelen sorun çoğu arabanın fiyat skalasının geniş olması ve buna bağlı olarak bir çok sınıfa dahil olduğu için 
#bir tahminde bulunmamız güçleşiyor.
#Bir diğer faktör dikkate alacağımız komşu sayısını yanlış seçmiş olma ihtimalimiz(5 tane olarak belirledik). Deneme yanılma 
#yaparak tahmin başarısını gözlemleyebilir ve seçimimizin ne derece doğru olduğunu bulabiliriz.
#ACC kesin olarak yüksek çıkmasını istiyorsak bütün arabaları tahminlemek yerine dahil oldukları sınıf sayısı daha az olan
#spesifik markaları seçip onlar arasında bir tahmin yapabiliriz. Sonuçlar daha iyi olacaktır.
print("ACC: ",accuracy_score(y_pred,y_test))
print(classification_report(y_test, y_pred)) 
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=2)  #komşu sayısını 3 yaptık.
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)  
from sklearn.metrics import classification_report, confusion_matrix  
cm=confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred)) 
#Az da olsa komşu sayısı ile oynayıp tahmin başarısını yükselttik. Yinede yetersiz.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score

x2=x[['price','height','make_peugot','make_volvo','make_subaru','make_alfa-romero','normalized-losses','wheel-base']]
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import precision_score

PS=0
for w in range (0,20):
    x2_train, x2_test, y_train, y_test = train_test_split(x2, y, test_size=0.20)
    scaler = MinMaxScaler()  
    scaler.fit(x2)
    x2_train = scaler.transform(x2_train)  
    x2_test = scaler.transform(x2_test) 

    clf = KNeighborsClassifier(n_neighbors = 3)
    cm=confusion_matrix(y_test, y_pred)
    classifier.fit(x2_train, y_train)
    y_pred = classifier.predict(x2_test)
    PS=precision_score(y_test, y_pred, average='macro')+PS
    
    
PS=PS/20
from sklearn.metrics import classification_report, confusion_matrix  
print(cm)  
print ('Average precision: %0.2f' % PS)
print(classification_report(y_test, y_pred)) 
#Burada modelimizi 20 kere çalıştırıp farklı eğitim ve test verileriyle elde ettiği accuracy değerlerini toplayıp ortalamasını
#aldık. K katmanlı çapraz doğrulama da kullanılabilirdi bunun için. 
#Görülüyor ki ilk sonuçlardan sonra belirttiğimiz gibi bir markanın fiyat skalasınınn geniş olması onun tahmin edilebilirliğini
#düşürüyor. Spesifik markalarla denediğimiz algoritmamız başarılı sayılabilecek sonuçlara ulaşmayı başardı.