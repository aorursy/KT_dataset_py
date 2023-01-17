import numpy as np

import seaborn as sns

import pandas as pd

df = pd.read_csv("/kaggle/input/iris/Iris.csv")
df.head()
df.shape
df.info()
df.describe()
df.isna().sum()
df.corr()
corr = df.corr()

sns.heatmap(corr,

           xticklabels=corr.columns.values,

           yticklabels=corr.columns.values);
df["Species"].unique()
df["Species"].mode()
sns.scatterplot(x= "SepalWidthCm", y="SepalLengthCm", data=df);
sns.jointplot(x= "SepalWidthCm", y="SepalLengthCm", data=df);
sns.scatterplot(x= "SepalWidthCm", y="SepalLengthCm", hue = "Species", data=df);
df.Species.value_counts()
sns.violinplot(y="SepalWidthCm", data = df)
sns.distplot(df["SepalWidthCm"], bins = 20, color="red")
sns.violinplot(y="SepalLengthCm", x="Species", data = df)
sns.countplot(x="Species", data=df)
sns.jointplot(x= "SepalLengthCm", y="SepalWidthCm", data=df);
sns.jointplot(x = df["SepalLengthCm"],y = df["SepalWidthCm"],kind="kde",color="black", data=df);
sns.scatterplot(x= "PetalWidthCm", y="PetalLengthCm", data=df);
sns.scatterplot(x= "PetalWidthCm", y="PetalLengthCm", hue = "Species", data=df);
sns.lmplot(x= "PetalLengthCm", y="PetalWidthCm", data=df);
df.corr()["PetalLengthCm"]["PetalWidthCm"]
df["TotalLengthCm"]=df["PetalLengthCm"].sum()+df["SepalLengthCm"].sum()
df.mean()["TotalLengthCm"]
df.std()["TotalLengthCm"]
df.max()["TotalLengthCm"]
df[(df["Species"]=="Iris-setosa") & (df["SepalLengthCm"]> 5.5)]
df[(df["Species"]=="Iris-virginica") & (df["PetalLengthCm"] < 5)].head()[["SepalLengthCm","SepalWidthCm"]]
df.groupby('Species').mean()
df.groupby('Species').std()[["PetalLengthCm"]]