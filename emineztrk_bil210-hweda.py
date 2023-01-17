import numpy as np
import seaborn as sns
import pandas as pd
df = pd.read_csv("../input/iriscsv/Iris.csv")
df.head()
df.shape
df.info()
df.mean()
df.describe().T
df.groupby(["variety"]).mean()
df.std()
df.groupby(["variety"]).std()
df.isna().sum() 
df.corr()
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values);
df["variety"].unique()
df["variety"].nunique()
sns.scatterplot(x = "sepal.width", y = "sepal.length", data = df);
sns.jointplot(x = "sepal.width", y = "sepal.length", data = df, color="crimson");
sns.scatterplot(x = "sepal.width", y = "sepal.length", hue = df["variety"], data = df);
df["variety"].value_counts()
sns.violinplot(y = "sepal.width", data = df);
sns.distplot(df["sepal.width"], bins=16, color="crimson");
sns.violinplot(df["variety"],df["sepal.length"])
sns.countplot(df["variety"])
sns.jointplot(x = df["sepal.length"], y = df["sepal.width"],color = "crimson");
sns.jointplot(x = df["sepal.length"], y = df["sepal.width"], kind = "kde", color = "crimson");
sns.scatterplot(x = df["petal.length"], y = df["petal.width"])
sns.scatterplot(x = df["petal.length"], y = df["petal.width"], hue =df["variety"])
sns.lmplot(x = "petal.length", y = "petal.width", data = df);
df.corr()["petal.length"]["petal.width"]
df["total.length"] = df["sepal.length"].sum() + df["petal.length"].sum()
print(df["total.length"])
df["total.length"].mean()
df["total.length"].std()
df["sepal.length"].max()
df[(df["variety"] == "Setosa") & (df["sepal.length"] > 5.5)]
df[(df["variety"] == "Virginica") & (df["petal.length"] < 5)][["sepal.length","sepal.width"]]
df.groupby(["variety"]).mean()
df.groupby(["variety"]).std()["petal.length"]