import numpy as np
import seaborn as sns
import pandas as pd
df = pd.read_csv("../input/iris.csv")

df.head()
df.shape
df.count
df.info()
df.describe().T
df.groupby(["variety"]).mean() # Ortalama olarak Virginica rengi çanakyaprağının uzunluğu daha büyük
df.groupby(["variety"]).std() # Standart sapma verilerine göre Virginica rengi çanakyaprağının uzunluğunun varyansı en büyük gözüküyor.
df.isna().sum() # Hiçbir öznitelikte eksik değer olmadığı görülüyor.




df.corr() #Korelasyon matrisi 4x4

df.corr()["petal.width"]["petal.length"] # En güçlü pozitif ilişki taçyaprağı uzunluğu ve genişliği arasında gözüküyor
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values); #Haritada gözüktüğü gibi sepal.width ve petal.lengh arasındaki ilişkinin koyu negatif 1'e daha yakın olduğunu görüyoruz.
df["variety"].unique()
df["variety"].nunique()
sns.scatterplot(x = "sepal.width", y = "sepal.length", data = df);

sns.jointplot(x = "sepal.width", y = "sepal.length", data = df, color="pink");


sns.scatterplot(x = "sepal.width", y = "sepal.length", hue = "variety", data = df); #Renkleri setosanın genişliği büyük, versicolorun ortalama, virginicanın uzunuluğu büyük şeklinde kümeleyebiliriz.
#Setosa geniş ve  kısa, Versicolor ortalama uzunlukta ve genişlikte, Virginica uzun ve ortalama genişlikte 
df.variety.value_counts()
sns.violinplot(x = "sepal.width", data = df); #Düzensiz bir dağılım var
sns.distplot(df["sepal.width"], bins=32, color="red");
sns.violinplot(x = "variety", y = "sepal.length", data = df);
sns.countplot(x = "variety", data = df);


sns.jointplot(x = df["sepal.length"], y = df["sepal.width"], color = "brown"); #Dağılım ve frekansın yüksek olduğu yer 6 ve 3 bölgeleri
sns.jointplot(x = df["sepal.length"], y = df["sepal.width"], kind = "kde", color = "brown");
sns.scatterplot(x = "petal.length", y = "petal.width",data = df);
sns.scatterplot(x = "petal.length", y = "petal.width", hue = "variety",  data = df);
sns.lmplot(x = "petal.length", y = "petal.width", data = df); #Aralarında pozitif doğrusal yönde güçlü bir ilişki var
df.corr()["petal.width"]["petal.length"]
df["total.length"]= df["sepal.length"].sum()+df["petal.length"].sum()

df["total.length"].mean()


df["total.length"].std()

df["sepal.length"].max()

df[(df["variety"] == "Setosa") & (df["sepal.length"] > 5.5)]

df[ (df["variety"] == "Virginica") & (df["petal.length"] < 5)][["sepal.length","sepal.width"]]

df.groupby(["variety"]).mean()  

df.groupby(["variety"]).std()["petal.length"]   
