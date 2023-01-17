import numpy as np

import seaborn as sns

import pandas as pd
df = pd.read_csv("../input/iris-dataset/iris_with_header.csv")
print(df.head(5))
df.shape
df.info()
df.describe()
df.isnull().sum()
df.corr()
corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
df["variety"].unique()
df["variety"].nunique()
df.plot.scatter(x='sepal.width', y='sepal.length' )
p = sns.jointplot(data=df,x='sepal.width', y='sepal.length')
ax = sns.scatterplot(x="sepal.width", y="sepal.length", hue="variety" ,data=df)
sns.violinplot(y = "sepal.width", data = df);
sns.distplot(df["sepal.width"], bins=16, color="darkblue");
sns.violinplot(x = "variety", y = "sepal.length", data = df);
ax = sns.countplot(x="variety", data=df)
sns.jointplot(x = "sepal.length", y = "sepal.width", data = df, color="darkblue");
sns.jointplot(x = "sepal.length", y = "sepal.width",kind="kde", data = df, color="darkblue");
sns.scatterplot(x = "petal.length", y = "petal.width", data = df);
sns.scatterplot(x = "petal.length", y = "petal.width", hue="variety", data = df);
sns.lmplot(x="petal.length", y="petal.width", data=df)
df.corr()["petal.length"]["petal.width"]
df['total.length'] = df['petal.length'] + df['sepal.length']

print(df.head())
df['total.length'].mean()
df['total.length'].std()
df['sepal.length'].max()
df[(df['variety']=='Setosa') & (df['sepal.length']>5.5)]
df.groupby('variety')['petal.length'].mean().head()
print(df.groupby("variety")["petal.length"].std().head())