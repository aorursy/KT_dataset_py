import numpy as np

import seaborn as sns

import pandas as pd
df = pd.read_csv("../input/iris.csv")

df.columns=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Species']



df.head()
df.shape
df.info()
df.describe().T
df.isnull()
df.corr()
corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
df["Species"].unique()
df["Species"].nunique()
sns.scatterplot(x='Sepal_Length',y='Sepal_Width',data=df);
sns.jointplot(df['Sepal_Length'],df['Sepal_Width'],data=df);
sns.scatterplot(x = "Sepal_Length", y = "Sepal_Width", hue = "Species" ,data = df);
df['Species'].value_counts()
sns.violinplot(y = "Sepal_Width", data = df); 
sns.distplot(df["Sepal_Width"], bins=16, color="blue");
sns.violinplot(x = "Species", y = "Sepal_Length",  data = df); 
sns.countplot(df['Species'])
sns.jointplot(x='Sepal_Length',y='Sepal_Width',color='green',data=df)
sns.jointplot(x='Sepal_Length',y='Sepal_Width',color='red',kind='kde',data=df)
sns.scatterplot(x='Petal_Length',y='Petal_Width',data=df)
sns.scatterplot(x='Petal_Length',y='Petal_Width',hue='Species',data=df)
sns.lmplot(x='Petal_Length',y='Petal_Length',data=df)
df.corr()["Petal_Length"]["Petal_Width"]
df['total_length'] = df['Petal_Length'] + df['Sepal_Length']
df['total_length'].mean()
df['total_length'].std()
df['total_length'].max()
df[(df["Species"] == "setosa") & (df["Sepal_Length"] > 5.5)]
df[(df['Petal_Length']<5) & (df['Species'] == "virginica")][["Sepal_Length","Sepal_Width"]] 
df.groupby('Species').mean()
grup=df.groupby('Species').mean()['Sepal_Length']

grup