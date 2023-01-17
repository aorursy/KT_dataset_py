import pandas as pd

import numpy as np

df = pd.read_csv("../input/diabetes/Diabetes.csv")

df.info()
df.head()
df = df.fillna(0)

df2 = df[["Glucose","BMI","Age","Outcome"]]
df2.head()
df2.describe()
#df3 = df2.loc[~(df2[df2.columns[:-1]] == 0).any(axis=1)]

#df3.describe()

#df3.info()

df3 = df2.loc[~(df2[df2.columns[:-1]]==0).any(axis=1)]

df3.describe()

df3.info()
df3.groupby("Outcome").mean()
df3.groupby("Outcome").agg({"Glucose": "mean", "BMI": "median", "Age": "sum"})
df3.groupby("Outcome").agg(["mean", "median"])
positive = df3.loc[df3["Outcome"] == 1]

negative = df3.loc[df3["Outcome"] == 0]

print(positive.shape, negative.shape)
df3.to_csv("clean_diabetes.csv", index=False)

df4  = pd.read_csv("./clean_diabetes.csv")

df4.head()

df4.info