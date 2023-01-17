import numpy as np
import seaborn as sns
import pandas as pd
df = pd.read_csv("iris.csv")
df.head()
df.shape
df.info()
df.describe()
df.isna().sum()
df.corr() #en güçlü pozitif ilişki petal.length ile petal.width arasındadır
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values);
df["variety"].unique()
df["variety"].nunique()
sns.scatterplot(x = "sepal.width", y = "sepal.length", data = df)
sns.jointplot(x = "sepal.width", y = "sepal.length", data = df ,color = "green");
sns.scatterplot(x = "sepal.width", y = "sepal.length" , hue ="variety" ,data = df);
df.count()
sns.violinplot(y = "sepal.width" ,data = df);
sns.distplot(df["sepal.width"] , bins = 16 , color = "blue");
sns.violinplot( x = "variety", y = "sepal.length" ,data = df);
sns.countplot(x = "variety", data = df);
sns.jointplot(x = "sepal.length", y = "sepal.width", data = df , color = "purple");
sns.jointplot(x = "sepal.length", y = "sepal.width", kind = "kde", data = df);
sns.scatterplot(x = "petal.length", y = "petal.width", data = df);
sns.scatterplot(x = "petal.length", y = "petal.width", hue = "variety", data = df);
sns.lmplot(x = "petal.length", y = "petal.width", data = df); #pozitif ve güçlü bir ilişki var
df.corr()["petal.length"]["petal.width"]
df["total.length"]=df["sepal.length"]+df["petal.length"]
df["total.length"]
df["total.length"].mean()
df["total.length"].std()
df["sepal.length"].max()
df[(df["sepal.length"]>5.5) & (df["variety"]=="Setosa")]
df[(df["petal.length"]<5) & (df["variety"]=="Setosa")]["sepal.length"]




df[(df["petal.length"]<5) & (df["variety"]=="Setosa")]["sepal.width"]
df.groupby(["variety"]).mean()
df.groupby(["variety"]).std()["petal.length"]