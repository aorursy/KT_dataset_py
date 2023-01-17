import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv("../input/DigiDB_digimonlist.csv")

df.head(3)
df.describe()
order = df.Type.value_counts()

order = order.sort_values()



plt.figure(figsize=(15, 8))

sns.countplot(df.Type, saturation=1, palette="Greens", order=order.index)
plt.figure(figsize=(20, 10))

df.Type.value_counts().plot(kind="bar")
table = df.Type.value_counts()

print(table.index)

labels = list(table.index)

print(table.values)

xdim = list(range(len(table)))

print(xdim)



plt.bar(xdim, table.values, tick_label=labels)