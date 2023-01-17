import pandas as pd
pd.set_option('display.max_columns', 500)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
dfBase = pd.read_csv('../input/Document.csv' #Read the Data Set From  CSV File || Otomobil.csv Dosyamızı Okuduk
                     , names=['symboling','normalized-losses','make','fuel-type'
                              ,'aspiration','num-of-doors','body-style','drive-wheels'
                              ,'engine-location','wheel-base','length','width','height'
                              ,'curb-weight','engine-type','num-of-cylinders','engine-size'
                              ,'fuel-system','bore','stroke','compression-ratio','horsepower'
                              ,'peak-rpm','city-mpg','highway-mpg','price'])
avg_mpg=((dfBase['city-mpg']+dfBase['highway-mpg'])/2)  
#Şehir İçi Ve Uzun Yol Yakıt Tüketimi İle Aracın Ortalama Yakıt Tüketimini Nitelik Olarak Türettik.
#Derived quality (city-mpg+highway-mpg)/2 equals average fuel consumption
dfBase.insert(25,'avg-mpg',avg_mpg)
dfBase.head()
dfBase.shape #rows and columns of data set  || veri setimizin satır ve sütun sayısını verir.
dfBase.head() #let's Take a look at the first 5 data || Verilerimize Bir Göz Atalım(Baştan 5 Adet'i)
dfBase.tail()#let's Take a look at the last 5 data || Verilerimize Bir Göz Atalım(Sondan 5 Adet'i)
dfBase.info()
#We see simple information  || Verimiz ile ilgili basit bir bilgi alalım
#Data Preprocessing || Veri Ön İşleme
df = dfBase.apply(lambda x: x.replace('?',np.NaN)) #global variable is assigned to free spaces || eksik değerlerimizi NaN Çevirelim
df.isnull().sum() #How many miss values we have || Eksik Değerlerimizin Dağılımını  Ve Hangi Kolonlarda Oldukları Görelim
df.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8) 
#Let's look at the distribution of our data and detect anomalies 
#Verilerimizin Dağılımına Bir Bakalım Ve Buradan Anomali Tespitlerini Yapabiliriz.    
df.describe() 
#Get basic information about data || Veri ile ilgili basit temel bilgiye göz atalım
df.corr() 
#Korelasyona Bakalım Korelasyon Projemize Ciddi Yön Verir. 
#Let's look at the correlation, the correlation is very important for our project

#Hangi Verilerin Bizim için daha kıymetli olduğunu,
#correlation shows which one data is so important for our
 
#Ayırt edici sütunları,veriler arasındaki ilişkinin oranını ve yönünü verir
#correlation shows the relationships and direction between the datas
data_corr=df.corr()
#Let's increase the visuality by drawing a heat map
#Isı Haritası Çizerek Görselliği Artıralım 
sns.heatmap(data_corr , vmax=.8, square=True) 
plt.title("Correlation of Data || Verilerin Koralesyonu") #Map Title || Haritanın Başlığı
plt.style.use('dark_background')
plt.show()
#We use the imputer class to fill in the missing values with the mean value
#Eksik Verilerimizi Ortalama Değerler İle Doldurmak İçin Sklearn Kütüphanesi Altından Imputer import ediyoruz
from sklearn.preprocessing import Imputer
#We will fill missing value with a mean || Eksik Verileri Ortalama Stratejisi Kullanarak Dolduracağız
imp = Imputer(missing_values='NaN', strategy='mean' )
df[['normalized-losses','bore','stroke','horsepower','peak-rpm','price']] = imp.fit_transform(df[['normalized-losses',
                                                                                                      'bore','stroke','horsepower','peak-rpm','price']])
df.head() 
df[df.isnull().any(axis=1)] #Is there another null value? || Eksik Verimiz Kalmış Mı Kontrol Ediyoruz
df["num-of-doors"].value_counts() #Let's fill the doors count with the most repetitive value
#NaN olan kapı sayılarını en çok tekrar eden değer ile dolduralım yani 4 İle
df['num-of-doors'] = df['num-of-doors'].fillna('four') #We have accepted the empty values as a 4-door car
#Eksik Değerlerimizi En Çok Tekrar Eden Değer İle Doldurduk
#we can show categorical columns as numerical data using dictionary for computer(algorithms and normalization)
#kategorik veriler sayısal verilere dönüştürülür bu işlem bilgisayar için yapılır(algoritmalar için ve normalizasyon için)
df['num-of-cylinders'].unique() #Non Repetitive Value || Tekrarsız Uniq Değerler
df['num-of-doors']=df['num-of-doors'].map({'two':2,'four':4}) #categorical datas convert numerical datas 
                                                              #kategorik verileri sayısal verilere dönüştürdük
df['num-of-cylinders']=df['num-of-cylinders'].map({'two':2,'three':3,'four':4,'five':5,'six':6, 'eight':8,'twelve':12})
df.head()
#Which car is the most sold? let's see!
#Araçların Satış Sayılarını görsel olarak görelim
plt.figure(figsize=(10, 5))
#plt.style.use('dark_background')
sns.countplot(x='make', data=df)
plt.xticks(rotation='vertical')
plt.title('Manufacturer')
plt.show()
for col in ['make','fuel-type','aspiration','body-style',
'drive-wheels','engine-location','engine-type','fuel-system']:
    df[col] = df[col].astype('category') #Veri Tipini katerogiye dönüştürüyoruz || Data Type Convert category

cat_df = df.select_dtypes(include=['category']).copy()
num_df= df.select_dtypes(exclude=['category']).copy() 

df.dtypes
cat_df.head()
num_df.head()
#(engine-size,curb-weight) and price have a positive high corelation.let's see
#motor hacmi,araç ağırlığı ile fiyat arasında pozitif yönlü ilişki mevcut bunları gösterelim
df_dum = pd.get_dummies(cat_df, columns= cat_df.columns)
num_df.corr()
sns.lmplot('price',"engine-size",data=df) #Bu İlişkiyi Görsel Olarak Görelim
#Let's see this relationship visually
sns.lmplot('price',"curb-weight",data=df) #Bu İlişkiyi Görsel Olarak Görelim
#Let's see this relationship visually
Q1 = df.quantile(0.25) #Çeyrek Değerleri alalım ve çeyrekler açıklığını Hesaplayalım Ve Böylece Uç Değerleri Bulup silebiliriz.
              #find quarter values and Let's calculate the Interquartile range so we can find and delete the outliers values

Q3 = df.quantile(0.75)
IQR = Q3-Q1
print(1.5*IQR)
print("Upper Outliers") #Upper Outliers || Üst Uç Değer Hesaplandı
print("------------------------")
print(Q1-IQR)
print("------------------------")
print("Lower Outliers") #Lower Outliers || Alt Uç Değer Hesaplandı   
print("------------------------")
print(Q3+IQR)
print("------------------------")
new_df = df[(df.price >=-926.00) & (df.price <=25186.00)]
new_df.shape
#17 Uç Değer Silindi 205 den 188 e düştü
#17 End Value Deleted from 205 to 188

sns.boxplot(x=new_df['price']) #Show Box Plot We see clearance of outliers || Boxplot Çizildi Uç Değerlerin Temizlendiğini Görüyoruz
#Kolonlardaki Değerlerimizin Sayısal Büyüklüklerinin,
 #Dağılımımızı Çarpıtmaması İçin Normalizasyon Yapıyoruz
#apply normalization to our values                                            
from sklearn.preprocessing import Normalizer 
numerical = num_df.values
normalizer = Normalizer().fit(numerical)
norm = normalizer.transform(numerical)
df_normalized = pd.DataFrame(norm)
df_normalized.columns = list(num_df.columns.values)
df_normalized.head()
dfMerged = pd.concat([df_dum,df_normalized,df_dum], axis=1)
dfMerged.head()
dfMerged.shape
prediction = dfMerged.iloc[204:205]
data = dfMerged.iloc[0:205]
#Linear Regresyon Modelimizi Uyguluyoruz Veri Kümemizi Test Train Olarak Bölüyoruz Ve Eğitiyoruz
#Ve K Katlamalı Çapraz Doğrulama Kullanıyoruz
#We are training data set with linear regression
#and use K Fold Cross Validation
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np

X = data.drop(['price'],axis=1).values 
Y = data['price'].values
X_train, X_test, y_train, y_test = train_test_split(X,Y)
model = LinearRegression(normalize=True)  
model.fit(X_train, y_train)  
y_pred = model.predict(X_test) #Predict Completed || Tahmin etme işlemi tamamlandı
from sklearn import metrics 
#let us measure the mean squared error of your estimates
#taminlerimiz ne kadar yaklaşık değer veriyor çeşitli metrikler ile buna bakalım
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  
    #Genel İşleyiş Ve Başarımın Gerekçelerinin Açıklanmas:
#Motor Hacmi-Ve(Ona Bağlı Olarak) Silindir Sayısı ile Fiyat Arasında Aynı Yönlü Ve Yüksek Oranlı Bir Korelasyon Bulunmakta.
#Buda Demek Oluyorki basitçe Biz Sadece Motor Hacmine Baksak Bile gerçeklikten çokta uzak olmayan sonuçlar verebiliriz.
#veri setimiz eğitim amaçlı kullanılan bir veri seti olduğu için bu şekilde değerler aldık ve gerçeğe yakın tahminler yapmak çok zor olmadı.
    #Data Preprocessing'in yorumlanması
#Eksik Değerlerimizi Ortalama İle Doldurmamız bizim için yeterli oldu fakat dahada güzel sonuçlar almak istersek,
#eksik verileri doldurmak için de ayrı bir regresyon modeli kullanabiliriz.
#kategorik bir takım verilerin çok az sayıda eksik değer içermesi sebebi ile çeşitli imputation yöntemleri kullanılabilirdi
#biz bazı değerlere direk ortalama ile doldurmayı seçerken bazılarına ise(genelde kategorik) en çok tekrar eden değeri,
#bir üstteki değeri direk almak gibi çeşitli imputation yöntemleri kullanmamız eksik veri sayımızın,
#genel veri sayısına oranla çok düşük olması sebebi ile sonucumuzu çok fazla değiştirmedi.
#outliers değerlerin çok fazla olması(17 adet) tahmin sapmamızı çok artırıyordu ve böylece sapmaya sebep oluyordu

