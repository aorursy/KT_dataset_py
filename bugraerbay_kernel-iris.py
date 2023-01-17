import numpy as np
import seaborn as sns
import pandas as pd
df = pd.read_csv("iris.csv")
df.head()
df.shape
df.info()
df.describe().T
df.isna().sum()
corr = df.corr()
corr
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values);
df["variety"].unique()
df["variety"].nunique()
sns.set(rc={'figure.figsize':(150,5)})
sns.jointplot(x = "sepal.width", y = "sepal.length", data = df, color="red");
sns.scatterplot(x = "sepal.width", y = "sepal.length", data = df, color = "red", hue = "variety");

sns.violinplot(y = "sepal.width", data = df);
sns.distplot(df["sepal.width"], bins=150, color="red");
sns.violinplot(y = "sepal.length",data = df);
df.count()
sns.jointplot(x = "sepal.length", y = "sepal.width", data = df, color="red");
sns.jointplot(x = "sepal.length", y = "sepal.width", data = df, kind = "kde", color="red");
sns.scatterplot(x = "petal.length", y = "petal.width",data = df);
sns.scatterplot(x = "petal.length", y = "petal.width",hue = "variety",data = df);
sns.lmplot(x = "petal.length",y = "petal.width",data = df)
df.corr()["petal.length"]["petal.width"]
df["total.length"] = pd.concat([df["petal.length"], df["sepal.length"]], axis = 1)


df.max()["sepal.length"]
df[(df["variety"]=="Setosa") & (df["sepal.length"]>5.5)]
df[(df["variety"]=="Virginica") & (df["petal.length"]<5)]["sepal.length"]
df.groupby(["variety"]).mean()
df.groupby(["variety"])["petal.length"].std()