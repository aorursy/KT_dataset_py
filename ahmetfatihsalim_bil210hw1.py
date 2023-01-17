import numpy as np
import seaborn as sns
import pandas as pd #kütüphaneleri ekledik
df = pd.read_csv("../input/irisdataset/iris.csv")
df.head()
df.shape#Gözlem ve nitelik
df.info()
df.describe().T
df.isna().sum()
df.corr()#peta.lenght ve peta.widht arası korelasyon katsayısı 1'e yakındır. bundan dolayı aralarındaki ilişki mükemmeldir.
sns.heatmap(df.corr()) 

df["variety"].unique()#dizi şeklinde döndürür
df["variety"].nunique()#adet bilgisini verir
sns.scatterplot(df['sepal.width'], df['sepal.length'])#kategorileştirmesiz, noktalı görselleştirme
sns.jointplot(x=df["sepal.width"],y=df["sepal.length"],color="red",kind="kde");
sns.scatterplot(x="sepal.width", y="sepal.length",hue="variety" , data=df) #viriginia ve versicolor'un sepal.lenght leri birbirine çok yakındır. bundan dolayı ayırt edilmeleri zordur.
df["variety"].value_counts()#tüm değişkenlerin dengeli dağıldığı görülebilir.
sns.violinplot(y="sepal.width",data=df);#Bu dağılım,normal dağılım değildir.
sns.distplot(df["sepal.width"],bins=16,color='green')
sns.violinplot(y = 'sepal.length', x = "variety", data = df)
ax = sns.countplot(x="variety", data=df)#Görselde ellişer Setosa,Versicolor ve Virginica türü çiçek bulunmaktadır.
g = sns.jointplot(x="sepal.length", y="sepal.width", data=df)#Yüksek frekansın  olduğu değer aralığı sepal.length'de ((5.5),(6.5)) aralığındayken sepal.width'de ((2.5), 3) aralığında
g = sns.jointplot(x="sepal.length", y="sepal.width",kind = "kde", data=df)
sns.scatterplot(x="petal.length",y="petal.width",data=df)
sns.scatterplot(x="petal.length",y="petal.width",hue = "variety",data=df)
sns.lmplot(x="petal.length",y="petal.width",data=df) #Petal.length ve petal.width arasında pozitif yönde , doğrusal ilişki vardır.
df.corr()["petal.width"]["petal.length"]
df["total.length"]= df["sepal.length"].sum()+df["petal.length"].sum()
print(df["total.length"])
df["total.length"].mean()
df["total.length"].std()
df.max()["petal.length"]
df[(df["sepal.length"] > 5.5 ) & (df["variety"] == "Setosa")]
df[(df["variety"] == "Virginica") & (df["petal.length"] < 5)][["sepal.length","sepal.width"]]
df.groupby(["variety"]).mean()
df.groupby(["variety"]).std()["petal.length"]