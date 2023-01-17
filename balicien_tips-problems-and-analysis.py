import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt 
tips=sns.load_dataset("tips")

df=tips.copy()
df.head()
df.describe().T
df.isnull().sum()
columns= df.columns

for column in columns:

    print(f"_____{column}______\n")

    print(df[column].value_counts())

    print("___________________\n")
plt.figure(figsize=(12,6))

sns.boxplot(data=df,x="day",y="tip",hue="time");
s=df.groupby(["day","smoker"])

s.sum()

s.mean()
plt.figure(figsize=(12,10))

sns.scatterplot(data=df,x="total_bill",y="tip",hue="sex");
plt.figure(figsize=(12,6))

sns.boxplot(data=df,x="smoker",y="tip",hue="time");