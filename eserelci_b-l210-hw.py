import numpy as np

import seaborn as sns

import pandas as pd
df = pd.read_csv("../input/iriscsv/iris (1).csv")
df.head()
df.shape
df.info()
df.groupby(["variety"]).std()
df.groupby(["variety"]).mean()
df.isna().sum()
df.corr()
corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
df["variety"].unique()
df["variety"].nunique()
sns.scatterplot(x = "sepal.width", y = "sepal.length", data = df);
sns.jointplot(x = "sepal.width", y = "sepal.length", data = df);
sns.scatterplot(x = "sepal.width", y = "sepal.length", hue = df["variety"], data = df);
df.variety.value_counts() #Görüldüğü üzere tüm değişkenler dengeli dağılmıştır.
sns.violinplot(x=df["sepal.width"]); #Dağılım,normal dağılım değildir.
sns.distplot(df["sepal.width"],bins=16, color="red");
sns.violinplot(df["variety"],df["sepal.length"])
sns.countplot(df["variety"]) #Görselde 50 adet Setosa,50 adet Versicolor,50 adet Virginica türü çiçek bulunmaktadır.
sns.jointplot(x=df["sepal.length"],y=df["sepal.width"])#Frekansın yüksek olduğu değer aralığı sepal.length'de 5.5 ve 6.5 aralığındayken sepal.width'de 2.5 3 aralığındadır.
sns.jointplot(x=df["sepal.length"],y=df["sepal.width"],kind="kde")
sns.scatterplot(x=df["petal.length"],y=df["petal.width"])
sns.scatterplot(x=df["petal.length"],y=df["petal.width"],hue=df["variety"])
sns.lmplot(x="petal.length",y="petal.width", data = df); #Petal.length ve petal.width aralarında pozitif yönde olmak üzere doğrusal bir ilişki vardır.
df.corr()["petal.length"]["petal.width"]
df["total.length"]= df["sepal.length"].sum()+df["petal.length"].sum()
print(df["total.length"])
df["total.length"].mean()
df["total.length"].std()
df["sepal.length"].max()
df[(df["variety"] == "Setosa") & (df["sepal.length"] > 5.5)]
df[(df["variety"] == "Virginica") & (df["petal.length"] < 5)][["sepal.length","sepal.width"]]
df.groupby(["variety"]).mean()
df.groupby(["variety"]).std()["petal.length"]