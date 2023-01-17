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
df["variety"].unique()
df["variety"].nunique()
sns.scatterplot(x = "sepal.width", y = "sepal.length", data = df);
sns.jointplot(x = "sepal.width", y = "sepal.length", data = df);
sns.scatterplot(x = "sepal.width", y = "sepal.length", hue="variety", data = df);
df["variety"].value_counts()

sns.violinplot(y="sepal.width", data=df)  #Değerler 3 üzerine yoğnlaşmış olup normal bir dağılım vardır. 

                                       
sns.distplot(df["sepal.width"])
sns.violinplot(y="sepal.length" , x="variety", data=df)
sns.countplot(x="variety" , data=df)
sns.jointplot(x="sepal.length" , y="sepal.width" , data=df)
sns.jointplot(x="sepal.length" , y="sepal.width" ,  kind = "kde" , data=df)
sns.scatterplot(x="petal.length" , y="petal.width" , data=df)
sns.scatterplot(x="petal.length" , y="petal.width" ,  hue = "variety" , data=df)
sns.lmplot(x = "petal.length", y = "petal.width" , data = df)
df.corr()["petal.length"]["petal.width"]  #Değer 1'e oldukça yakındır. Pozitif ve güçlü bir ilişki vardır.
df["total.length"] = df["petal.length"].add(df["sepal.length"])
df["total.length"].mean() 
df["total.length"].std()
df["sepal.length"].max()
df[(df['sepal.length'] > 5.5) & (df['variety'] == "Setosa")]
df[(df["petal.length"] < 5.0 ) & (df["variety"] == "Virginica")][["sepal.length","sepal.width"]]
df.groupby(["variety"]).mean()
df.groupby(df["variety"])[["petal.length"]].std()