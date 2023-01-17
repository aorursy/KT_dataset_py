import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from IPython.display import display

import seaborn as sns

print(os.listdir("../input"))
df1 = pd.read_csv("../input/rock.csv")

df1.head(10)
df1['area'].mean()
df2 = pd.read_csv("../input/trees.csv")

df2.head(10)
df2['Height'].mean()
df3 = pd.read_csv("../input/titanic.csv")

df3.head(10)
df3['Age'].std()
df2 = pd.read_csv("../input/trees.csv")

df2.head(10)
print(df2['Girth'].median())

print(df2['Girth'].describe())

Ans = 12.9-11.05

print(Ans)
df1 = pd.read_csv("../input/rock.csv")

df1.head(10)
df4 = df1[["area", "shape"]]

print(df4)

print("-----------------------------------")

print("Correlation:\n", df4.corr())