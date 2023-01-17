import numpy as np
import seaborn as sns
import pandas as pd
df = pd.read_csv("iris.csv")
df.head()
df.shape
df.info
df.describe().T
df.isna().sum()
df.corr()
corr = df.corr()
sns.heatmap(corr,
           xticklabels=corr.columns.values,
           yticklabels=corr.columns.values);
df["variety"].unique()
df["variety"].nunique()
sns.scatterplot(x="sepal.width", y="sepal.length", data = df);
sns.jointplot(x="sepal.width", y="sepal.length", data = df , color="orange");
sns.scatterplot(x = "sepal.length", y = "sepal.width",hue="variety",data=df, color="blue")
sl = df['sepal.length'].value_counts(normalize=False)
sw = df['sepal.width'].value_counts(normalize=False)
pl = df['petal.length'].value_counts(normalize=False)
pw = df['petal.width'].value_counts(normalize=False)
print(sl, sw, pl, pw)
sns.violinplot(y="sepal.width", data=df);
sns.distplot(df["sepal.width"], bins=16, color="blue");
sns.violinplot(x = "sepal.length", y = "variety", data=df, )
sns.countplot(x="variety", data=df)
sns.jointplot(x="sepal.length", y="sepal.width", data=df)
sns.jointplot(x= "sepal.length", y="sepal.width", kind="kde", data=df)
sns.scatterplot(x="petal.length", y="petal.width", data=df)
sns.scatterplot(x="petal.length", y="petal.width",hue="variety", data=df)
sns.lmplot(x = "petal.length", y = "petal.width", data = df);
df.corr()['petal.length']['petal.width']
total_length = df['petal.length'].add(df['petal.width'])
total_length.mean()
total_length.std()
df['sepal.length'].max()
df[(df['sepal.length']>5) & (df['variety'] == "Setosa")]
df[(df['petal.length']<5) & (df['variety']=="Virginica")].filter(["sepal.length", "sepal.width"])
df.groupby(['variety']).mean()
df.groupby(['variety'])['petal.length'].std()

