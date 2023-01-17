import numpy as np
import seaborn as sns
import pandas as pd
df = pd.read_csv("../input/iriscsv/iris.csv")
df.head()
df.shape
df.info()
df["sepal.length"].describe().T 
df["sepal.width"].describe().T
df["petal.length"].describe().T
df["petal.width"].describe().T
df.isna().sum() 
df.corr() #En güçlü pozitif ilişki total.length ile petal.length arasında olduğu görülüyor.
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values);
df["variety"].unique()
df["variety"].unique()
sns.scatterplot(x = "sepal.width", y = "sepal.length", data = df);
sns.jointplot(x = "sepal.width", y = "sepal.length", data = df, color="orange");
sns.scatterplot(x = "sepal.width", y = "sepal.length", hue="variety", data = df);
df['variety'].value_counts()
sns.violinplot(y = "sepal.width", data = df);
sns.distplot(df["sepal.width"], bins=16, color="orange");
sns.violinplot(y = 'sepal.length', x = "variety", data = df)
sns.countplot(df['variety'])
sns.jointplot(x = "sepal.width", y = "sepal.length", data = df, color="orange");
sns.jointplot(x = "sepal.width", y = "sepal.length", kind="kde",data = df, color="orange");
sns.scatterplot(x = "petal.width", y = "petal.length", data = df);
sns.scatterplot(x = "sepal.width", y = "sepal.length", hue="variety", data = df);
sns.lmplot(x = "petal.length", y = "petal.width", data = df);
df.corr()["petal.width"]["petal.length"]
df['total.length'] = df['sepal.length'] + df['petal.length']
df
df["total.length"].mean() 
df["total.length"].std() 
df["sepal.length"].max() 
df[(df["sepal.length"] > 5.5 ) & (df["variety"] == "Setosa")]
df[(df["sepal.length"] < 5.0 ) & (df["variety"] == "Virginica")][["sepal.length","sepal.width"]]
df.groupby(["variety"]).mean()
df.groupby(["variety"]).std().head()["petal.length"]