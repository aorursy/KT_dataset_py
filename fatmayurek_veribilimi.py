import numpy as np
import seaborn as sns
import pandas as pd

df = pd.read_csv("../input/irisdataset/iris.xls")
df.head(5)

df.shape
df.info()
df.describe().T
#eksik gözlemleri göz ardı eder.
#kategorik değişkenleri ele almaz.
df.mean()
#Ortalamaya bakıldığında;
#"sepal.length" değişkeni en büyük,"petal.width" değişkeni en küçük ortalamaya sahiptir.
df.std()
#Standart sapmaya bakıldığında ;
#"petal.length" değişkeninin std 'si en büyük,
#"sepal.width" değişkeninin std 'si en küçüktür.
#Eğer birçok veri ortalamaya yakın ise Standart Sapma’da düşük olacaktır.

#Bu duruma bakıldığında "sepal.width"en düzenli, "pedal.length" en düzensiz dağıldığı söylenebilir.
#Varyans,verilerin ortalamadan sapmasının kareleri toplamının ortalamasıdır.
#Yani standart sapmanın karekök alınmamış halidir.
# Bu durumda;"petal.length" in varyansı en yüksek,"sepal.width" in varyansı en düşüktür ,sonucunu çıkarırız.

#Kontrol edelim:
df.var()
df.isna().sum()
df.corr()
#Korelasyon katsayısı +1.00 a yaklaştıkça değişkenler arasında pozitif ilişki olur.
#Bu nedenle en güçlü pozitif ilişki "petal.length" ve "petal.width" arasındadır.
corr = df.corr()
sns.heatmap(corr,annot=True ,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values);

df["variety"].unique()
df["variety"].nunique()
sns.scatterplot(x="sepal.width",y="sepal.length",data=df,color="orange");
sns.jointplot(x = "sepal.width", y = "sepal.length", data = df, color="purple");
sns.pairplot(df,hue="variety");
#pairplot=jointplotun tüm değişkenler için yapılmış halidir
sns.scatterplot(x = "sepal.width", y = "sepal.length",hue="variety",style="variety",data=df);
df["variety"].value_counts() 
#Değişkenler ve değerleri eşit olarak dağıldığı için dengeli bir dağılıma sahiptir.
#3 değişken türümüz ve toplam 150 değerimiz olduğunu biliyoruz.Bu durumda hepsinin 50 tane değer içermesi dengelidir.
sns.violinplot(x="sepal.width",data=df);
#Grafiğe bakıldığında;3.0 aralığında dağılım artarken başa ve sona giderken azalmış.
#Normal dağılımdır.Normal dağılım denilmesinin nedeni ise doğada her türlü verinin bu şekilde dağılış göstermesidir.
#Örneğin;boy uzunlukları veya ağırlıkları cinsiyete uygun olarak dağılmaktadır. Ortalama bir boy veya ağırlık en fazla sayıda bulunurken aşırı küçük veya büyük değerler bu ortalamadan uzaklaştıkça azalacaktır.
sns.distplot(df["sepal.width"],color="purple");
sns.violinplot(x=df["variety"],y=df["sepal.length"],data=df);
sns.countplot(x="variety",data=df);
sns.jointplot(x="sepal.length",y="sepal.width",data=df,color="yellow");
sns.jointplot(x="sepal.length",y="sepal.width",data=df,kind="hex",color="red");
#'Koyu renkli alanlarda yüksek dağılım vardır' diyebiliriz.
sns.jointplot(x="sepal.length",y="sepal.width",kind="kde",data=df);
sns.scatterplot(x="petal.length",y="petal.width",data=df,color="green");
sns.scatterplot(x="petal.length",y="petal.width",hue="variety",data=df);
sns.lmplot(x="petal.length",y="petal.width",data=df);
#Korelasyon sonucunda aralarında pozitif ve güçlü bir ilişki olduğunu göstermiştik.
#Bu grafiğe bakıldığında doğrusal yönde bir grafik ve değişkenler arası güçlü bir ilişki olduğunu söyleyebiliriz.

df[["petal.length","petal.width"]].corr()


df["total.length"]=df[["sepal.length","petal.length"]].sum(axis=1)
df.shape #Yeni bir column eklemiş olduk.

df
df["total.length"].mean()
df["total.length"].std()
df["total.length"].max()
df[(df["variety"] == "Setosa") & (df["sepal.length"] > 5.5)]
df_filtered = df[(df["variety"] == "Virginica")&(df["petal.length"] < 5) ][["sepal.length", "sepal.width"]]

df_filtered
df.groupby(["variety"]).mean()
df.groupby(["variety"]).std()["petal.length"]