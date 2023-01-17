import numpy as np

import seaborn as sns

import pandas as pd
df = pd.read_csv("../input/datasetofiris/iris.csv")
df.head(5)
df.shape
df.info()
df.describe()
df.isna().sum()
df.corr()
corr = df.corr()

sns.heatmap(corr,

           xticklabels=corr.columns.values,

           yticklabels=corr.columns.values);
df["variety"].unique()
df["variety"].nunique()
sns.scatterplot(x="sepal.width", y = "sepal.length",data=df);
sns.jointplot(x=df["sepal.width"], y =df["sepal.length"],color= "red");
sns.scatterplot(x="sepal.width",y="sepal.length",hue="variety", data=df)
df["variety"].value_counts()
sns.violinplot(y="sepal.width",data=df);
sns.distplot(df["sepal.width"],bins=16,color="red");
sns.violinplot(y = "sepal.length", x = "variety", data = df);
sns.countplot(x = "variety", data = df);
sns.jointplot(x=df["sepal.width"], y =df["sepal.length"],color= "red");
sns.jointplot(x=df["sepal.width"], y =df["sepal.length"],kind="kde",color= "red");
sns.scatterplot(x="petal.width", y = "petal.length",data=df);
sns.scatterplot(x="petal.width", y = "petal.length",hue="variety",data=df);
sns.lmplot(x="petal.length",y="petal.width",data=df); #güçlü bir ilişki vardır
df.corr()["petal.length"]["petal.width"]
df["total.length"] = df[['petal.length','sepal.length']].sum(axis=1)

df
df["total.length"].mean(axis=0)
df["total.length"].std(axis=0)
df["sepal.length"].max()
df[(df["variety"] == "Setosa") & (df["sepal.length"] > 5.5)]
df[(df["variety"] == "Virginica") & (df["petal.length"] < 5)].head()[["sepal.length","sepal.width"]]
df.groupby(["variety"]).mean()
(df.groupby(["variety"])["petal.length"]).std()