import numpy as np
import seaborn as sns
import pandas as pd


df = pd.read_csv("../input/irisdataset/iris.csv")
df.head()
df.shape
df.info()
df.std()
#varyans standart sapmanın karesi alınmış halidir.
df.isna().sum()
df.corr().T
#petal.length ile petal.witdh arasında pozitif yönde en güçlü ilişki vardır.
corr=df.corr().T
sns.heatmap(corr,
           xticklabels=corr.columns.values,
           yticklabels=corr.columns.values)
df["variety"].unique()
df["variety"].nunique()
sns.scatterplot(x="sepal.width",y="sepal.length", data=df);
sns.jointplot(x="sepal.width",y="sepal.length", data=df,color ="black");
sns.scatterplot(x="sepal.width",y="sepal.length",hue="variety", data=df)
#turuncu versicolor mavi setosa yeşil virginicadır.
df["variety"].value_counts()
#veriler çok dengelidir
sns.violinplot(y="sepal.width",data=df)
sns.distplot(df["sepal.width"],bins=16,color="black");
sns.violinplot(y="sepal.length",x="variety",data=df)
sns.countplot(x="variety",data=df);
sns.jointplot(x="sepal.width",y="sepal.length", data=df,color ="red");
#sepal.length 5,5-6,5 arası ve sepal.width 2,5-3 arası yoğunluk vardır
sns.jointplot(x="sepal.width",y="sepal.length",kind="kde", data=df,color ="red");

sns.scatterplot(x="petal.width",y="petal.length", data=df)
sns.scatterplot(x="petal.width",y="petal.length",hue="variety", data=df)
sns.lmplot(x="petal.length",y="petal.width",data=df,hue="variety");
#grafikteki noktalardan düz bir çizgi çizebildiğimiz için güçlü bir ilişki vardır
df.corr()["petal.length"]["petal.width"]
df["total.length"] = df[['petal.length','sepal.length']].sum(axis=1)
df
df["total.length"].mean(axis=0)
df["total.length"].std(axis=0)
df["sepal.length"].max()
df[(df["sepal.length"] > 5.5) & (df["variety"] == "Setosa")]
df[(df["petal.length"] < 5) & (df["variety"] == "Virginica")][["sepal.length","sepal.width"]]
df.groupby("variety").apply(lambda x:np.mean(x))
(df.groupby(["variety"])["petal.length"]).std()