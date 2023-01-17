import numpy as np
import seaborn as sns
import pandas as pd
df = pd.read_csv("../input/seaborn-iris-dataset/iris.csv")
df.head()
df.shape
df.info
df.describe().T
df.std()
df.mean()
df.isna().sum()
df.corr()
corr=df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values);
df["variety"].unique()
df["variety"].nunique()
sns.scatterplot(x = "sepal.width", y = "sepal.length", data = df)
sns.jointplot(x="sepal.width" , y = "sepal.length" , data = df, color ="violet")
sns.scatterplot(x = "sepal.width" , y = "sepal.length" , data = df, hue = "variety")
df["variety"].value_counts()
sns.violinplot(x = "sepal.width",data=df)
sns.distplot(df["sepal.width"],bins = 10, color = "blue")
sns.violinplot(y = "sepal.length", x = "variety", data = df , variety = ["setosa" , "virginica" , "versicolor" ])
sns.countplot(df["variety"])
sns.jointplot(x= "sepal.length" , y = "sepal.width", data = df)
sns.jointplot(x = "sepal.length" , y = "sepal.width", data = df, kind = "kde", color="yellow")
sns.scatterplot(x = "petal.length", y = "petal.width", data = df, color = "orange")
sns.scatterplot(x = "petal.length", y = "petal.width", hue= "variety", data = df)
sns.lmplot(x = "petal.length", y = "petal.width", data = df)
df.corr()["petal.length"]["petal.width"]
df["total.length"]=df["petal.length"].sum()+df["sepal.length"].sum()
df["total.length"].mean()
df["total.length"].std()
df["sepal.length"].max()
df[(df["sepal.length"]>5.5) & (df["variety" ] == "Setosa")]
(df[(df["petal.length"] < 5) & (df["variety"] == "Virginica")])[["sepal.length","sepal.width"]]
df.groupby("variety").mean()
df.groupby("variety")["petal.length"].std()