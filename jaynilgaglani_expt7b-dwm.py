from sklearn.cluster import DBSCAN

from sklearn.preprocessing import StandardScaler

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/wholesale-customers-data-set/Wholesale customers data.csv");

data1 = df

print(df.head())
print(df.info())
print(df.describe())
df.drop(["Channel", "Region"], axis = 1, inplace = True)
print(df.head())
x = df['Grocery']

y = df['Milk']



plt.scatter(x,y)

plt.xlabel("Groceries")

plt.ylabel("Milk")

plt.show()
df = df[["Grocery", "Milk"]]
stscaler = StandardScaler().fit(df)

df = stscaler.transform(df)
dbsc = DBSCAN(eps = .5, min_samples = 15).fit(df)
labels = dbsc.labels_

core_samples = np.zeros_like(labels, dtype = bool)

core_samples[dbsc.core_sample_indices_] = True
import seaborn as sns



filtro=list(core_samples)



data1["Filtro"]=filtro



sns.lmplot("Grocery","Milk",data=data1,fit_reg=False,hue="Filtro",size=10)