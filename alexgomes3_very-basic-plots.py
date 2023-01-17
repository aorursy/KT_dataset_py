import os

import pandas as pd

import numpy as np

import seaborn as sns; sns.set()

import matplotlib.pyplot as plt

%matplotlib inline

from mpl_toolkits.mplot3d import Axes3D

os.listdir('.')
df = pd.read_csv('../input/train.csv')

corrmat=df.corr()

plt.figure(figsize=(20,10),dpi=96)

sns.heatmap(corrmat, vmin=0, vmax=1)
df2=pd.DataFrame(corrmat["SalePrice"])

df2.sort_values("SalePrice",ascending=False)
plt.figure(1)

plt.scatter(df["OverallQual"],df["SalePrice"])

plt.figure(2)

plt.scatter(df["GrLivArea"],df["SalePrice"])
plt.scatter(df["GrLivArea"],df["SalePrice"])
fig = plt.figure(3)

ax = Axes3D(fig)

ax.scatter(df["OverallQual"],

           df["GrLivArea"],

           df["SalePrice"])



plt.show()