import pandas as pd 
import numpy as np
import seaborn as sns
iris_dataset=pd.read_csv("../input/iris-dataset/iris.csv")
iris_dataset
iris_dataset.shape
iris_dataset.ndim
iris_dataset.size
iris_dataset.describe
df_iris=pd.DataFrame(iris_dataset)
df_iris
df_iris.info()
df_iris.count()
df_iris.isna().sum()
df_iris.ndim
df_iris.shape
df_iris.axes
df_iris.values
type(df_iris)
df_iris.index
df_iris.head()#eğer fonksiyonun içine herhangi bir değer girmezsek dataframedeki ilk 5 satırı getirir
df_iris.head(10)#fonksiyona yazdığımız değer kadar baştan başlayarak satırı getirir
df_iris.tail()# fonksiyonun içine bir şey yazmazsak sondan 5 satırı getirir
df_iris[0:15]#bu kullanım için head ve tail fonksiyonuna benzer diyebiliriz ancak bu kullanımla başlangıcı ve bitiş değerlerini biz belirlemiş oluyoruz.Bu kullanım yerine df_iris.head(15) bize aynı sonucu verir
df_iris[10:30]
df_iris.loc[3]
df_iris.iloc[3]
df_iris
df_iris.drop(0,axis=0)#bir satır silmek istediğimiz zaman drop fonksiyonu kullanırız.Fonksiyona girilen ilk parametre index ismi olmalı.Burada girmiş olduğum axis parametresi ise bize silme 
#işlemini satır veya sütun bazlı mı olucağını belirtir.axis değeri 0 ikeN satır bazlı 1 iken sütun bazlı işlem yapılır.
df_iris
#Az önce 0 index isimli satırı silmiştik ancak gördüğümüz gibi tekrar dataframe incelendiğinde 0 indexi hala gözükmekte.Bunun sebebi drop fonksiyonu inplace isimli bool döndüren bir parametre daha alması.
#Biz bu parametreyi True olarak belirtmediğimiz  zaman default olarak False olarak çalışır ve silme işlemi asıl dataframede silinmez
df_iris2=pd.DataFrame(iris_dataset)
df_iris2
df_iris2.drop(0,axis=0,inplace=True)#bir dataframe daha oluşturup inplace fonksiyonun nası çalıştığını göstermek istedim
df_iris2
df_iris2.iloc[5]#burda 5 değerini yazdık ama name değişkenine baktığımız zaman 6 numaraları satırı getirmiş çünkü 0,1,2,3,4,5 numaralı olarak indexleri gezdi.5.index de 6 isimli veri satırına ait.
df_iris2.loc[5]#loc ise direk istediğimiz satırı getirdi.5 isimli index olduğu için 5 indexli satırı getiri
df_iris2.iloc[0:3]#0.indexten başlayarak 3.indexe kadar seçim yaptım
#2.indexteki satırın 2.indexteki sutunun kesitğini değeri alalım
df_iris2.iloc[2,2]
#indexlerin yazılı olduğun sütunun bir indexi yok burdaki 0.index ilk isimlendirilen değişken ile başlıyor.
#2.indexten 5.indexe kadar satırları ve aynı şekilde sütunları aldım
df_iris2.iloc[2:5,2:5]
df_iris2.iloc[3:6]["petal.length"]#istenilen sutünlar arasında petal.length değişkenine göz attım
df_iris#şimdi ana dataframe üzerinde çalışmaya devam edelim
# drop işlemini şu şekilde de yapabiliriz
silinecek_satirlar=[2,5]
df_iris.drop(silinecek_satirlar,axis=0)
#gördüğünüz gibi 2 ve 5 numaralı satırlar silinmiş oldu
#verilerimizi küçükten büyüğe sıralayalım
df_iris.sort_values("sepal.length",ascending=True)
#değişken isimlerinin olup olmadığını kontrol etmek istiyorum
df_iris.head()
"petal.length" in df_iris
aranan_sutunlar=["sepal.length","petal.width","petal"]
for i in aranan_sutunlar:
    print(i in df_iris)
#petal isimli bir değişkenin olmadığını gördük
#herhangi bir değişken eklemek istiyorum
df_iris["exampleVariable"]=df_iris["sepal.length"]*df_iris["sepal.width"]
df_iris.head()
df_iris.drop("exampleVariable",axis=1,inplace=True)#sutun bazlı silme işlemi yapıcağım için axis=1 olarak aldım ve değişikliğin kalıcı olması için inplace=True olarak değiştirdim
df_iris.head()
df_iris[0:2]["petal.length"]
#yukarıdaki verileri dataframe şeklinde görmek için değişkeni bir köşeli parantezin içine daha almalıyız.
df_iris[0:2][["petal.length"]]

#sepal.length değişkeninin 5den daha büyük olduğu değerleri görmek istiyorum
df_iris[df_iris["sepal.length"]>5]#ve kaç adet olduğunu öğrenmek istiyorum
df_iris[df_iris["sepal.length"]>5].shape
#şimdi de sepal.length değişkenin 5 den büyük ve sepal.width değişkenin 3.5dan büyük değerlerini görüntülemek istiyorum
df_iris[(df_iris["sepal.length"]>5)&(df_iris["sepal.width"]>3.5)]
df_iris[(df_iris["sepal.length"]>5)&(df_iris["sepal.width"]>3.5)].shape#isteğimize uyan ne kadar veri olduğunu görelim
#tek tek değişkenlerin ortalamalarını öğrenmek istiyorum
df_iris.mean()
#petal.length değişkeni üzerinde çalışalım
df_iris["petal.length"].mean()#ortalamasını alalım
df_iris["variety"].nunique()
df_iris["petal.length"].count()#sayısını öğrenelim
df_iris["petal.length"].min()
df_iris["petal.length"].max()
df_iris["petal.length"].sum()#toplamlarımı öğrenelim
df_iris["petal.length"].var()#varyansını alalım
df_iris["petal.length"].std()
df_iris.describe()#yukarıda yaptığımız işlemleri tek bir tabloda her değişken için görelim
df_iris.describe().T#Transpozunu alıp daha güzel bir görünüm elde edelim
#datasetimi  variety değişkenime göre gruplandırmak istiyorum
df_iris.groupby("variety")
df_iris.groupby("variety").mean()#variety değişkenine göre gruplandırıp ortalamasını aldım
df_iris.groupby("variety").sum()
df_iris.groupby("variety")["sepal.length"].mean()#variety değişkenine göre gruplandırmıştık şimdi de gruplara göre bir sepal.length değişkenin ortalamasını aldım
df_iris.groupby("variety")["petal.length"].std()
df_iris.groupby("variety")["sepal.length"].describe()
#Şimdi de variety değişkenine göre gruplandırdığımız verisetimizin değişkenlerinin minumum,maximum ve medyan değerlerini bulup kümeleyelim
df_iris.groupby("variety").aggregate([min,np.median,max])
#Filtreleme işlemi yapmak için bir fonksiyon kullanmak istiyorum
def filter_func(x):
    return x["sepal.length"].std()>0.5
#Şimdi bu fonksiyonu kullanarak sepal.length değişkeni üzerinde bir filtreleme işleme yapalım
df_iris.groupby("variety").filter(filter_func)
#Her bir değişkende değişiklik yapmak için transform işlemi yapmak istiyorum
df_iris2=df_iris.iloc[:,1:3]
df_iris2
#Her bir değişken değerlerinden o değişkenin ortalamasını çıkarıp değişkenin standart sapmasına böldüm
df_iris2.transform(lambda x:(x-x.mean())/x.std())
#Yukarıdaki işlemi bir fonksiyon yazarak şöyle de yapabilirdik
def function(x):
    return (x-x.mean())/x.std()
df_iris2.transform(function)
#apply fonksiyonu kullanarak ortalama ve toplam hesaplayalım
df_iris.apply(np.sum)
df_iris.apply([np.sum,np.mean])#dataframe içinde ve birlikte gözlemleyelim]

#Variety edğişkenindeki nan değerini silmek için dropna fonskiyonunu kullanalım 
df_clear=df_iris.apply([np.sum,np.mean])
df_clear.dropna(axis=1)
#ancak bu çok sağlıklı bir yöntem değil çünkü veri kaybına yol açtık zaten variety değişkenlerinin toplamına ihtiyacımız yoktu ama daha farklı datasetlerde büyük veri kaybına yol açabilirz
df_iris.groupby("variety")[["sepal.length","sepal.width"]].aggregate("mean")
df_iris.pivot_table(["sepal.length","sepal.width"],index="variety")
import matplotlib.pyplot as plt
import seaborn as sns
fig=plt.figure(figsize=(6,4))
axes=fig.add_axes([0,0,1,1])
axes.plot(df_iris["sepal.length"],color="red",label="Sepal Length")
axes.plot(df_iris["sepal.width"],color="purple",label="Sepal Width")
axes.plot(df_iris["petal.length"],color="yellow",label="Petal Length")
axes.plot(df_iris["petal.width"],color="black",label="Petal Width")
axes.legend()
df_iris.cov()
df_iris.corr()
corr = df_iris.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values);
sns.scatterplot(x = "sepal.length", y = "sepal.width", data = df_iris);
sns.jointplot(x = "sepal.length", y = "sepal.width", data = df_iris, color="purple");
sns.scatterplot(x = "sepal.length", y = "sepal.width", hue = "variety",  data = df_iris);
pd.value_counts(df_iris.values.flatten())
sns.violinplot(y = "sepal.width", data = df_iris);
sns.distplot(df_iris["sepal.width"], bins=16, color="purple");
sns.violinplot(x="variety",y = "sepal.width", data = df_iris);
sns.countplot(x = "variety", data = df_iris);
sns.jointplot(x = df_iris["sepal.length"], y = df_iris["sepal.width"], color = "purple");
sns.jointplot(x = df_iris["sepal.length"], y = df_iris["sepal.width"],kind="kde", color = "purple");
sns.scatterplot(x = "petal.length", y = "petal.width",  data = df_iris);
sns.scatterplot(x = "petal.length", y = "petal.width",hue="variety", data = df_iris);
sns.lmplot(x = "petal.length", y = "petal.width", data = df_iris);
df_iris[["petal.length","petal.width"]].corr()
df_iris["total.length"]=df_iris["petal.length"]+df_iris["sepal.length"]
df_iris
df_iris["total.length"].mean()
df_iris["total.length"].std()
df_iris["sepal.length"].max()
df_iris[(df_iris["sepal.length"]>5.5)&(df_iris["variety"]=="Setosa")]
df_iris[(df_iris["petal.length"]<5)&(df_iris["variety"]=="Virginica")][["sepal.length","sepal.width"]]

