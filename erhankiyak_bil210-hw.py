import numpy as np

import seaborn as sns

import pandas as pd
df = pd.read_csv("../input/irisdataset/iris.csv")
df.head()
df.shape
df.info()
df.describe().T
df.isna().sum()
df.corr()
sns.heatmap(df.corr())
df['variety'].unique()
df['variety'].nunique()
sns.scatterplot(df['sepal.width'], df['sepal.length'])
sns.jointplot(df['sepal.width'], df['sepal.length'])
sns.scatterplot(df['sepal.width'], df['sepal.length'], hue = df['variety'])
df['variety'].value_counts()
sns.violinplot(y = "sepal.width", data = df)
sns.distplot(df["sepal.width"])
sns.violinplot(y = 'sepal.length', x = "variety", data = df)
sns.countplot(df['variety'])
sns.jointplot(df['sepal.length'],df['sepal.width'])
sns.jointplot(df['sepal.length'],df['sepal.width'], kind = 'kde')
sns.scatterplot(df['sepal.length'],df['sepal.width'])
sns.scatterplot(df['sepal.length'], df['sepal.width'], hue = df['variety'])
sns.lmplot(x = "petal.length", y = "petal.width" , data = df)
df.corr()["petal.length"]["petal.width"]
df["total.length"] = df['petal.length'] + df['sepal.length']

total_length = df['total.length']
total_length.mean()
total_length.std()
total_length.max()
df[(df["sepal.length"] > 5.5 ) & (df["variety"] == "Setosa")]
df[(df["sepal.length"] < 5.0 ) & (df["variety"] == "Virginica")][["sepal.length","sepal.width"]]
df.groupby(["variety"]).mean()
df.groupby(df["variety"])[["petal.length"]].std()