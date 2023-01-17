import numpy as np
import seaborn as sns
import pandas as pd
df = pd.read_csv("../input/practisingondatasets-with-seaborn-python-library/iris.csv")
df.head()
df.shape
df.dtypes
df["sepal.length"].std()**2
df["sepal.length"].mean()
df.isna().sum()
df.corr()
df.corr()["petal.length"]["sepal.length"]
sns.set(rc={'figure.figsize':(8,6)})
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values);
df["variety"]
df["variety"].unique()
sns.scatterplot(x = "sepal.width", y = "sepal.length", data = df);
sns.jointplot(x = df["sepal.width"], y = df["sepal.length"], kind = "kde", color = "purple");
sns.scatterplot(x = "sepal.width", y = "sepal.length", hue = "variety",  data = df);
idx = pd.Index(df["variety"])
idx.value_counts()
sns.violinplot(y = "sepal.width", data = df);
sns.distplot(df["sepal.width"], bins=16, color="purple");
sns.violinplot(x = "sepal.length", y = "variety", data = df);
sns.countplot(df['variety'])
sns.jointplot(x = df["sepal.width"], y = df["sepal.length"], color = "blue");
sns.jointplot(x = df["sepal.width"], y = df["sepal.length"], kind= "kde", color = "blue");
sns.scatterplot(x = df["petal.length"], y = df["petal.width"], color = "blue");
sns.scatterplot(x = df["petal.length"], y = df["petal.width"], hue = df["variety"], color = "blue");
sns.lmplot(x = "petal.length", y = "petal.width", data= df);
df.corr()["petal.length"]["petal.width"]
df["total.length"] = df["petal.length"] + df["sepal.length"]
df["total.length"].mean()
df["total.length"].std()
max(df["sepal.length"])
df[(df["sepal.length"] > 5.5) & (df["variety"] == "Setosa")]
df[(df["petal.length"] < 5) & (df["variety"] == "Virginica")][["sepal.length", "sepal.width"]]
df.groupby(["variety"]).mean()
df.groupby(["variety"])["petal.length"].std()