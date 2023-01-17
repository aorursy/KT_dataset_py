import numpy as np

import seaborn as sns

import pandas as pd
df = pd.read_csv("../input/iris-dataset/iris.csv")
df.head()
df.shape # veriseti 150 gözlem  ve 5 sütundan oluşuyor.
df.info()
df.describe().T # standart sapmalara baktığımızda düşük olduğunu görüyoruz. 

#Verilerin aralarında çok fark olmadığını ve dengeli dağıldığını gösterir.
df.isna().sum() # boş değer içeren gözlem yok.
df.corr()#korelasyon değeri 1 e yaklaştıkça aralarındaki ilişki güçlü olur. 

#Buradaki verilerin arasında güçlü bir ilişki olduğunu görüyoruz.

#petal.width ve petal.length değişkenleri arasında güçlü bir ilişki var.
corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);

# Isı haritasına bakınca aralarındaki ilişkinin nasıl oldığunu daha iyi görüyoruz.

#Genel olarak bakıldığında değişkenler aralarında güçlü bir ilişki var.
df["variety"].unique()
df["variety"].nunique()
sns.scatterplot(x = "sepal.width", y = "sepal.length", data = df);
sns.jointplot(x = "sepal.width", y = "sepal.length", data = df, color="crimson");
sns.scatterplot(x = "sepal.width", y = "sepal.length", hue = df["variety"], data = df);
df["variety"].value_counts() #Tüm variety ler dengeli bir şekilde dağılmış.
sns.violinplot(y = "sepal.width", data = df); #Normal dağılım değildir.
sns.distplot(df["sepal.width"], bins=16, color="crimson");
sns.violinplot(df["variety"],df["sepal.length"])
sns.countplot(df["variety"])# Grafik bize 50 adet setosa , 50 adet versicolor ve 50 adet virginica olduğunu gösterir.
sns.jointplot(x = df["sepal.length"], y = df["sepal.width"],color = "crimson");

#sepal.width 3.0 ile 3.5 arasında, sepal.length 5.5 ve 7.0 arasında frekans yüksektir.
sns.jointplot(x = df["sepal.length"], y = df["sepal.width"], kind = "kde", color = "crimson");

#grafikte koyu renkli yerler bize frekansın yüksek olduğu yerleri gösterir.
sns.scatterplot(x = df["petal.length"], y = df["petal.width"]) 

# grafiğe baktığımızda lineer bire dağılım görüyoruz.
sns.scatterplot(x = df["petal.length"], y = df["petal.width"], hue =df["variety"])
sns.lmplot(x = "petal.length", y = "petal.width", data = df);

#petal.length ve petal.width değişkenleri arasında lineer bir ilişki var ve bu aralarındakki ilişkinin güçlü olduğunu gösterir. 
df.corr()["petal.length"]["petal.width"]# korelasyon katsayısı 1 e oldukça yakın ve korelasyon 1 e yaklaştıkça aralarındaki ilişkinin güçlü oldupunu gösterir.
df["total.length"] = df["sepal.length"].sum() + df["petal.length"].sum()
print(df["total.length"])
df["total.length"].mean()
df["total.length"].std() # standart sapmasını düşük sayalım. standart sapma artıkça değerler arasındaki farkk fazla olur. 
df["sepal.length"].max()
df[(df["variety"] == "Setosa") & (df["sepal.length"] > 5.5)] # bu özelliği sağlayan 3 tane gözlem var.
df[(df["variety"] == "Virginica") & (df["petal.length"] < 5)][["sepal.length","sepal.width"]]
df.groupby(["variety"]).mean()
df.groupby(["variety"]).std()["petal.length"] #standart sapmalar çok düşük.dengeli bir dağılım  var.