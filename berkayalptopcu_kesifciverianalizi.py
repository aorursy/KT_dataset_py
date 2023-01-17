import numpy as np

import seaborn as sns

import pandas as pd
df = pd.read_csv("../input/iris-dataset/iris_with_header.csv")
df.head()
df.shape
df.info()
df.describe().T
df.isna( ).sum( )
df.corr()
corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
df["variety"].unique()
df["variety"].nunique()
sns.scatterplot(x = "sepal.width", y = "sepal.length", data = df);
sns.jointplot(x = "sepal.width", y = "sepal.length", data = df, color="green");
sns.scatterplot(x = "sepal.width", y = "sepal.length",hue="variety", data = df);
df["variety"].value_counts()
sns.violinplot(y = "sepal.width", data = df);
sns.distplot(df["sepal.width"], bins=16, color="green");
sns.violinplot(x = "variety", y = "sepal.width", data = df);
sns.countplot(x = "variety", data = df);
sns.jointplot(x = df["sepal.length"], y = df["sepal.width"], color = "orange");
sns.jointplot(x = df["sepal.length"], y = df["sepal.width"], kind = "kde", color = "orange");
sns.scatterplot(x = "petal.length", y = "petal.width", data = df);
sns.scatterplot(x = "petal.length", y = "petal.width", hue = "variety", data = df);
sns.lmplot(x="petal.length", y="petal.width", data=df)
df.corr()["petal.width"]["petal.length"]
df['total.length'] = df['petal.length'] + df['sepal.length']

print(df.head())
df['total.length'].mean()
df['total.length'].std()
df['sepal.length'].max()
df[(df['variety']=='Setosa') & (df['sepal.length']>5.5)]
df[(df['variety']=='Virginica') & (df['petal.length']<5)][["sepal.width","sepal.length"]]
df.groupby('variety')['petal.length'].mean().head()
print(df.groupby("variety")["petal.length"].std().head())