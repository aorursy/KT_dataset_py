import numpy as np
import seaborn as sns
import pandas as pd
df = pd.read_csv("../input/iris-dataset/iris.csv")
df.head()
df.count()
df.info()
df.mean()
df.std()
df.isna().sum()
df.corr()
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values);
df["variety"].unique()
df["variety"].nunique()
sns.scatterplot(x="sepal.width", y="sepal.length", data=df, color="magenta");
sns.jointplot(x="sepal.width", y="sepal.length", data=df, color="magenta");
sns.scatterplot(x="sepal.width", y="sepal.length", hue="variety", data=df);
pd.value_counts(df.values.flatten())
sns.violinplot(y = "sepal.width", data = df);
sns.distplot(df["sepal.width"], bins=16, color="blue");
sns.violinplot(x = "variety", y = "sepal.length", data = df);
sns.countplot(x = "variety", data = df);
sns.jointplot(x = df["sepal.width"], y = df["sepal.length"], color = "purple");
sns.jointplot(x = df["sepal.width"], y = df["sepal.length"],kind="kde", color = "purple");
sns.scatterplot(x="petal.width", y="petal.length", data=df);
sns.scatterplot(x="petal.width", y="petal.length",hue="variety", data=df);
sns.lmplot(x="petal.width", y="petal.length", data=df);
df[['petal.length', 'petal.width']].corr()
df['total.length'] = df['petal.length'] + df['sepal.length']
df['total.length'].mean()
df['total.length'].std()
df['sepal.length'].max()
df[(df['variety']=='Setosa') & (df['sepal.length']>5.5)]
df[(df['variety']=='Virginica') & (df['petal.length']<5)][['sepal.length','sepal.width']]
df.groupby(["variety"]).mean()