import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings 

warnings.filterwarnings("ignore")
df=pd.read_csv("../input/Automobile_data.csv")
pd.set_option("max_columns",40)
df.head()
df.shape
df.tail()
df.describe()
df.describe(include="object")
df.isnull().sum()
nul=df["price"].isnull()==True
df.loc[nul]
df["price"].nunique()
df.loc[df["company"]=="isuzu"]
df.loc[df["company"]=="porsche"]
df["price"].dtype
df.loc[df["company"]=="toyota"]
df.company.value_counts()
df.groupby("company")["price"].max().sort_values(ascending=False)
df.groupby("company")["average-mileage"].mean().sort_values(ascending=False)
df.sort_values("price",ascending=False)
sns.heatmap(df.corr(),annot=True)
plt.plot(df["price"],"ro--",c="green")

plt.legend()
np.log(df["horsepower"]).hist()
plt.scatter(df["horsepower"],df["price"])

plt.subplot()
df["horsepower"].plot()
for j in df:

    if df[j].dtype!="O":

        sns.barplot(df[j],df["price"])

        plt.title(j)

        plt.show()

        

for col in df.columns:

    if df[col].dtype=="O":

        sns.scatterplot(df["price"],df[col])

        plt.title(col)

        plt.show()
df.loc[df["price"]==df["price"].max()]