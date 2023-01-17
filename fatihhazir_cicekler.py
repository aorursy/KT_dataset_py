import numpy as np

import seaborn as sns

import pandas as pd
df = pd.read_csv("../input/cicekler/iris.csv")
df.head(5)
df.shape # 150 gozlem ve 5 öznitelik.
df.info()
df.describe() # Sepal length özniteliğine baktığımzda standart sapmasının düşük oldugunu goruyoruz. Demek oluyor ki veriler ortalama etrafında kümelenmiş.Sepal width özniteliğine

#baktıgımızda ise aynı şekilde std degeri düşük. Demek ki onda da veri kümelenmesi ortalama deger etrafında. Petal length kısmına geldiğimzde ise bizi yüksek bir std degeri karşılıyor.

#Buradan anlayabiliriz ki verilerin kümelenmesi ortalama etrafında değil. Uç verilerden bahsedebiliriz. Son olarak petal width özniteliğine baktıgımızda biraz daha sakin bir dagılım

#gorüyoruz. Veriler dogrudan ortalama etrafında kümelenmiş diyemeyiz ama ortalamadan da çok uzaklaşmamışlar.
df.isna().sum().sum()
df.corr() # Degerleri incelediğimizde en yüksek pozitif ilişkinin petal length ile petal width degerleri arasında oldugunu goruyoruz. Neredeyse 1'e yakın degere sahipler.
corr = df.corr()

sns.heatmap(corr,

           xticklabels=corr.columns.values,

           yticklabels=corr.columns.values)
df["variety"].unique()
df["variety"].nunique()
sns.scatterplot(x = "sepal.width", y = "sepal.length", data = df);
sns.jointplot(x = "sepal.width", y = "sepal.length", data = df, color = "orange")
sp = sns.scatterplot(x = "sepal.width", y = "sepal.length", hue = "variety" ,data = df, color = "orange" ) # Veri sayımız az iken 3 farklı renk ile kumeleme yapılabilir. Yine veri sayımız

#az iken ayırt edilebilir duruyor. Fakat veri sayısı çogaldıgında çok da sağlıklı olmayacaktır.
df["variety"].value_counts()  #Oldukca dengeli bir dagılım söz konusu.
sns.violinplot(y = "sepal.width", data = df) # Normale yaklaşan bir grafige sahip. Ortalaması üçten biraz daha düşük olsa normali yakalayabilirdi.
sns.distplot(df["sepal.width"], bins=16, color = "blue");
sns.violinplot(x = df["variety"], y = "sepal.length", data = df)
sns.countplot(x = "variety", data = df)
sns.jointplot(x = "sepal.length", y = "sepal.width", data = df, color = "purple")
sns.jointplot(x = "sepal.length", y = "sepal.width", data = df, color = "purple", kind = "kde")
sns.scatterplot(x = "petal.length", y = "petal.width", data = df)
sns.scatterplot(x = "petal.length", y = "petal.width", data = df, hue="variety")
sns.lmplot(x = "petal.length", y = "petal.width", data = df) # Grafige bakılırsa aralarında kesinlikle dogru orantı oldugunu söyleyebiliriz. Hatta korelasyon kat sayısını da hatırlarsak

#birbirlerini pozitif yönde desteklediklerini soylemek yanlış olmaz. Aralarındaki ilişki güçlüdür.
df.corr()["petal.length"]["petal.width"]
totalLength = df["sepal.length"] + df["petal.length"]
totalLength.mean()
totalLength.std()
df["sepal.length"].head(1)
df[(df["sepal.length"] > 5.5) & (df["variety"] == "Setosa")]
df[(df["petal.length"] < 5) & (df["variety"] == "Virginica")][["sepal.length","sepal.width"]]
df.groupby(["variety"]).mean()
df.groupby(["variety"])["petal.length"].std()