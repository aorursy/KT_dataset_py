import numpy as np

import seaborn as sns

import pandas as pd
df = pd.read_csv("../input/iriscsviris/iris.csv")
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
sns.violinplot(x=df["sepal.width"]); #Dağılım görüldüğü üzere normal dağılım değildir.
sns.distplot(df["sepal.width"],bins=16, color="red");
sns.violinplot(df["variety"],df["sepal.length"])
sns.countplot(df["variety"]) #Görselde görüldüğü üzere her çiçekten ellişer tane bulunmaktadır.
sns.jointplot(x=df["sepal.length"],y=df["sepal.width"])
sns.jointplot(x=df["sepal.length"],y=df["sepal.width"],kind="kde")
sns.scatterplot(x=df["petal.length"],y=df["petal.width"])
sns.scatterplot(x=df["petal.length"],y=df["petal.width"],hue=df["variety"])
sns.scatterplot(x=df["petal.length"],y=df["petal.width"],hue=df["variety"])  

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