import numpy as np
import seaborn as sns
import pandas as pd
import os

df = pd.read_csv("../input/iris/Iris.csv")
df.tail()
df.shape
df.info()
df.describe( ).T
df.isna( ).sum( ) 
df.corr()
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values);
df["Species"].unique()
df["Species"].nunique()
sns.scatterplot(x = "SepalWidthCm", y = "SepalLengthCm", data = df);
sns.jointplot(x = "SepalLengthCm", y = "SepalWidthCm", data = df);
sns.scatterplot(x = "SepalWidthCm", y = "SepalLengthCm",hue="Species", data = df);
df["Species"].value_counts()
sns.violinplot(y = "SepalWidthCm", data = df);
sns.distplot(df["SepalWidthCm"]);
sns.violinplot(x = "Species", y = "SepalWidthCm", data = df);
sns.countplot(x = "Species", data = df);
sns.jointplot(x = df["SepalLengthCm"], y = df["SepalWidthCm"]);
sns.jointplot(x = df["SepalLengthCm"], y = df["SepalWidthCm"], kind = "kde");
sns.scatterplot(x = "PetalLengthCm", y = "PetalWidthCm", data = df);
sns.scatterplot(x = "PetalLengthCm", y = "PetalWidthCm", hue = "Species", data = df);
sns.lmplot(x="PetalLengthCm", y="PetalWidthCm", data=df)
df.corr()["PetalWidthCm"]["PetalLengthCm"]
df['TotalLengthCm'] = df['PetalLengthCm'] + df['SepalLengthCm']
print(df.head())
df['TotalLengthCm'].mean()
df['TotalLengthCm'].std()
df['SepalLengthCm'].max()
df[(df['Species']=='Iris-setosa') & (df['SepalLengthCm']>5.5)]
df[(df['Species']=='Iris-virginica') & (df['PetalLengthCm']<5)][["SepalWidthCm","SepalLengthCm"]]
df.groupby('Species')['PetalLengthCm'].mean().head()
print(df.groupby("Species")["PetalLengthCm"].std().head())