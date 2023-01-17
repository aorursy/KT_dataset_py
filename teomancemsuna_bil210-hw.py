import numpy as np
import seaborn as sns
import pandas as pd
df = pd.read_csv("../input/dataset/iris.csv")
df.head()
df.shape
df.info()
df.describe()
df.isna().sum()#Eksik değer yok
df.corr()
# petal.width ve petal.lenght değişkenleri arasındaki korelasyon katsayi değeri 1'e en yakin olduğundan ve + yönde olduğundan en güçlü pozitif ilişkidir.
corr=df.corr()
sns.heatmap(corr,xticklabels=corr.columns.values,
                 yticklabels=corr.columns.values);
df["variety"]
df["variety"].nunique() 
sns.scatterplot(x="sepal.width",y="sepal.length",data=df);
sns.jointplot(x="sepal.width",y="sepal.length",data=df);
sns.scatterplot(x="sepal.width",y="sepal.length",hue="variety",data=df);
#Ayırt edicilik niteliklerin çeşitlendirilmesiyle artar.Ancak veriler düzensiz bir şekilde dağıldıysa okunurluk azalabilir.
df["variety"].value_counts()
sns.violinplot(y="sepal.width",data=df);
sns.distplot(df["sepal.width"],bins=16,color="black");
sns.violinplot(y="sepal.width",x="variety",data=df);
sns.countplot(x="variety",data=df)
sns.jointplot(x="sepal.length",y="sepal.width",data=df);
sns.jointplot(x="sepal.length",y="sepal.width",kind="kde",data=df);
sns.scatterplot(x="petal.length",y="petal.width",color="purple",data=df);
sns.scatterplot(x="petal.length",y="petal.width",hue="variety",data=df);
sns.lmplot(x="petal.length",y="petal.width",data=df);
# Genişlik ve uzunluk doğru orantılıdır. Buda pozitif yönde biri artarken diğerininde arttığı anlamına gelir.
df.corr()["petal.length"]["petal.width"]
df["total.length"] = df["petal.length"]+df["sepal.length"]
df["total.length"].mean()
df["total.length"].std()
df["sepal.length"].max()
df[(df["sepal.length"]>5.5) & (df["variety"]=="Setosa")]
df[(df["petal.length"]<5)& (df["variety"]=="Virginica")].filter(["sepal.length","sepal.width"]) 
#İlk olarak df icerisine coklu sorgulamalar yaptirildi donen degerlerden olusan df ise istenen degisken degerlerine gore 
#filter fonksiyonuyla filtrelendi.
df.groupby(["variety"]).mean()
df.groupby("variety").std()["petal.length"]