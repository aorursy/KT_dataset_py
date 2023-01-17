import numpy as np
import seaborn as sns
import pandas as pd
df = pd.read_csv("../input/sonirisdataset/iris.csv")
df.head(5)
df.shape
df.info()
#Değişken tipleri ve bellek kullanımı
df.describe()
#varyansın karesi alındığında standart sapma elde edilir.
df.isna().sum()
#eksik değer bu datasetinde mevcut değildir.
df.corr()
#Korelasyon katsayıları 1 ile -1 arasında değer alır.
#En güçlü pozitif ilişki  petal.length ile petal.width arasındadır ve bu ikisinin katsayısı 0.962865dir.
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values);
#Korelasyon katsayılarının ısı hartası
df["variety"].unique()
#Hedef değişkenlerimiz
df["variety"].nunique()
#Hedef değişkenlerimizin sayısı
sns.scatterplot(x="sepal.width", y="sepal.length", data=df, color="magenta");
#sepal.width ve sepal.length değişkenlerinin süreklilik grafiği
sns.jointplot(x="sepal.width", y="sepal.length", data= df, color="red");
#Bir satırdaki veya sütundaki verilerin yoğunlaşmasını bu grafik analiz etmemizi sağlar
sns.scatterplot(x= "sepal.width" , y = "sepal.length", hue = "variety", data=df);
#Mavi renkteki noktalar, turuncu ve yeşil renkteki noktalardan bağımsızdır fakat çokta fark yoktur.
#Bundan dolayı kümeleme işlemi kullanılabilir.
df["variety"].value_counts()
#Her bir değerli değişkenin kaç defa kullanıldığını gösterir.
#Göründüğü üzere oldukça dengeli bir dağılım vardır.
sns.violinplot(y="sepal.width", data=df);
#3 e yaklaştıkça değer sayımızın artmakta olduğu söylenebilir.
#Normal bir dağılımdır.
sns.distplot(df["sepal.width"], color="blue");
sns.violinplot(x= df["variety"], y= df["sepal.length"], data=df);
sns.countplot(x = "variety", data=df);
sns.jointplot(x = "sepal.length", y = "sepal.width", data = df, color="green");
#sepal.width 3 noktasında, sepal.length 5 ve 6-6,5 noktaları arasındaki dağılımların frekansları yüksektir.
sns.jointplot(x = "sepal.length", y = "sepal.width", data=df, kind= "kde", color="purple");
sns.scatterplot(x = "petal.length", y = "petal.width", data = df);
sns.scatterplot(x = 'petal.length', y = 'petal.width', hue = 'variety', data = df);

sns.lmplot(x="petal.length", y= "petal.width", data=df);
#Pozitif yönde, doğrusal ve güçlü bir ilişki vardır
df[["petal.width","petal.length"]].corr()
df["total.length"] = df[['petal.length','sepal.length']].sum(axis = 1)
df
df["total.length"].mean(axis=0)
df["total.length"].std(axis=0)
df["total.length"].max()
df[(df["sepal.length"]> 5.5) & (df["variety"] == "Setosa")]
df[(df["petal.length"] < 5) & (df["variety"] == "Virginica")][["sepal.length","sepal.width"]]
df.groupby("variety").apply(lambda x:np.mean(x))
(df.groupby(["variety"])["petal.length"]).std()