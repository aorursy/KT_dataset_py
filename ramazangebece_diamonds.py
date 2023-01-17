import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
diamonds=sns.load_dataset('diamonds')
df=diamonds.copy()
df.head()
#53940 tane gözlem birimi ve 10 tane değişken var.
df.shape
df.info()
df.dtypes
df.describe().T
df.carat.describe().T
df_carat=pd.cut(df.carat,[0.2,1,2,3,4,5])
df_carat
#'carat' adlı değişken 5 sınıfa ayrıldı,örneğin 0.2-1.0 aralığında 36426 tane değer var.
df_carat.value_counts()
sns.countplot(df_carat)
plt.show()
#şimdi bu aralıkların 'price' adlı değişkeninin etkileme durumunu grafik ile inceleylim:
#soru:acaba 'carat' değeri arttıkça 'price' değeri de artıyor mu?
#aşağıdaki grafikte,örneğin 4.0-5.0 aralığındaki 'price' değeri o sınıfın ortalama değeridir.
sns.barplot(x=df_carat,y='price',data=df)
plt.show()
#her bir sınıfın ortalama fiyat değerlerini nümerik olarak ta hesaplayabiliz.
df.groupby(df_carat)['price'].mean()
df.cut.head()
#'cut' adlı değişkenimizin sınıflarını frekanslarına erişelim:
df.cut.value_counts()
from pandas.api.types import CategoricalDtype
#şimdi 'cut' adlı kategorik değişkenimizi,ordinal kategorik değişken olarak ayarlayalım:
df.cut=df.cut.astype(CategoricalDtype(categories=['Fair','Good','Very Good','Premium','Ideal'],ordered=True))
df.dtypes
df.cut.head(5)
df.cut.value_counts()
sns.countplot(df.cut)
plt.show()
(df.cut.value_counts()
 .plot
 .barh()
 .set_title('cut değişkeninin sınıflarının frekansları'))
plt.show()
#soru:'cut' adlı kategorik değişkendeki sınfların 'price' adlı değişkeni etkileme durumu nedir?
#her bir sınıfın ortalama fiyatını görsel ve nümerik olarak inceliyelim:
#dikkat:en yüksek ortalam fiyat Premium sınıfında gözüküyor.
sns.barplot(x='cut',y='price',data=df)
plt.show()
#nümrerik olarak gördüğümüz gibi,en yüksek ortalama fiyat.Premium sınıfında
df.groupby('cut')['price'].mean()
#soru:'cut' adlı kategorik değişkenin sınıflarının herbirindeki 'color' adlı değişkene göre ortalama 'price' değerleri:
#örneğin,Fair adlı snıfın d adlı renk değerindeki ortalama 'price' değeri 4291 ..gibi nümerik çıkarımlar elde edebiliriz.
df.groupby(['cut','color'])['price'].mean()
#görsel olarak ta gözlemleyebiliriz:
sns.barplot(x='cut',y='price',hue='color',data=df)
plt.show()
df.head()
df.price.describe().T
df_price=pd.cut(df.price,[0,2500,5000,7500,10000,12500,17500])
df_price.head()
df_price.value_counts()
#dikkat:sürekli değişken,kategorik değişkene dönüştürüldü ve 5 sınıfa bölündü.
#grafiğer bakarak;'price' adlı değişkenin,0-2500 değerleri aralığında yaklaşık olarak 25.000 in üzerinde gözlem var.
plt.figure(figsize=(10,7))
sns.countplot(df_price)
plt.show()
#dikkat:'price' adlı sürekli değişkenin,gözlemlerinin belirli değer aralıklarında ki toplam değer sayısı 
#histogram grafiği.
plt.figure(figsize=(10,7))
sns.distplot(df.price,bins=10,kde=False)
plt.show()
#'price' adlı bağımlı değişkenimzin yoğunluk grafiği:
#soru:bu yoğunluk grafiğinde,yoğunluğun 0-5000 arasında fazla olmasının sebebi nedir?
#acaba hangi bağımsız değişken bu kısmı etkiliyor?
#bu soruların cevaplarını da çaprazlama yöntemi ile analiz edebiliriz.
plt.figure(figsize=(10,7))
sns.kdeplot(df.price,shade=True)
plt.show()
sns.FacetGrid(df,hue='cut',height=5,
              xlim=(0,10000)).map(sns.kdeplot,'price',shade=True).add_legend()
plt.show();
df.head(3)
plt.figure(figsize=(10,7))
sns.scatterplot(x='price',y='carat',data=df)
plt.show();
df_carat.value_counts()
#soru:'carat' adlı değişende 0.2-1 aralığındaki değerlerin ortalama 'price' değerini bulalım.
#ilk önce karşılaştırma operatörleri ile bu değer aralığında olan değerler True olarak döndü.
#yani True değerlerimiz bu değerler aralığında.
(df.carat>0.2) & (df.carat<1)
#buradaki işlemde,eğer bir pandas ve ya numpy ın içeriisine bir True-False işlemi gönderirsek,bize sadece True olan değerler gelir.
#dikkat!,burada sadece bu aralıklar içindeki değerler var.
df[(df.carat>0.2) & (df.carat<1)]
#şimdi ilk başta yazdığımz sorumuza dönelim,bu değerler aralığındaki 'price' değişkeninin ortalama değeri kaçtır?
#sonuç:'carat' değişkeninin 0.2-1.0 aralığındaki değerlerinin(toplam değer:36426 tane),'price' değişkenindeki ortalama değer:1633.07..
df[(df.carat>0.2) & (df.carat<1)]['price'].mean()
plt.figure(figsize=(10,7))
sns.scatterplot(x='price',y='carat',hue='cut',style='color',data=df)
plt.show();
df.groupby(['cut','color'])['price','carat'].mean()
