import numpy as np

import seaborn as sns

import pandas as pd
df = pd.read_csv("../input/irisdataset/iris.csv")
df.head()
df.tail()
df.sample(5)
df.shape #(Gözlem,Nitelik) 
df.info()
df.describe()
pow(df.std(),2) #Varyans değerlerini Standart Sapmalarının karelerini alarak buldum.
df.isna().sum()
df.corr()
corr=df.corr()

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
df["variety"].unique()
df["variety"].nunique()
sns.scatterplot(x="sepal.width",y="sepal.length",data=df);
sns.jointplot(x="sepal.width",y="sepal.length",data=df);
sns.scatterplot(x="sepal.width",y="sepal.length",data=df,hue="variety");
df["variety"].value_counts()
sns.violinplot(y = "sepal.width", data = df);
sns.distplot(df["sepal.width"], bins=16 ,color="darkblue");
sns.violinplot(x = "variety", y = "sepal.length", data = df);
sns.countplot(x="variety",data=df)
sns.jointplot(x = df["sepal.length"], y = df["sepal.width"], color = "darkblue");
sns.jointplot(x = df["sepal.length"], y = df["sepal.width"], kind = "kde", color = "red");
sns.scatterplot(x="petal.width",y="petal.length",data=df);
sns.scatterplot(x="petal.width",y="petal.length",data=df,hue="variety");
sns.lmplot(x = "petal.width", y = "petal.length", data = df);
df.corr()["petal.width"]["petal.length"]
df["Total.Length"] = df["petal.length"].add(df["sepal.length"])
df["Total.Length"].mean()
df["Total.Length"].std()
df["sepal.length"].max()
df[(df['sepal.length']>5.5) & (df['variety'] == "Setosa")]
df[(df['petal.length']<5) & (df['variety'] == "Virginica")][["sepal.length", "sepal.width"]]
df.groupby(["variety"]).mean()
df.groupby(["variety"]).std()["petal.length"]