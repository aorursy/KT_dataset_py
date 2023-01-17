import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go



df = pd.read_csv("../input/mr-boston-flattened.csv")

print('Categories',df['category'].nunique())
print(df['category'].value_counts()[:11].sort_values())
print('Total number of unique ingredients 1:',df['ingredient-1'].nunique())
print('Total number of unique ingredients 2:',df['ingredient-2'].nunique())
#print('Total number of unique ingredients 3:',df['ingredient-3'].nunique())
#print('Total number of unique ingredients 4:',df['ingredient-4'].nunique())
#print('Total number of unique ingredients 5:',df['ingredient-5'].nunique())
#print('Total number of unique ingredients 6:',df['ingredient-6'].nunique())

fig, ax=plt.subplots(figsize=(16,7))

df['category'].value_counts().sort_values(ascending=False).head(11).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2)

plt.xlabel('Category',fontsize=20)
plt.ylabel('Number',fontsize=20)
ax.tick_params(labelsize=20)
plt.title('11 Categories',fontsize=25)
plt.grid()
plt.ioff()

fig, ax=plt.subplots(figsize=(16,7))

df['ingredient-1'].value_counts().sort_values(ascending=False).head(50).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2)
#df['ingredient-2'].value_counts().sort_values(ascending=False).head(30).plot.bar(width=0.5,edgecolor='r',align='center',linewidth=2)
#df['ingredient-3'].value_counts().sort_values(ascending=False).head(30).plot.bar(width=0.5,edgecolor='g',align='center',linewidth=2)
#df['ingredient-4'].value_counts().sort_values(ascending=False).head(30).plot.bar(width=0.5,edgecolor='b',align='center',linewidth=2)
#df['ingredient-5'].value_counts().sort_values(ascending=False).head(30).plot.bar(width=0.5,edgecolor='m',align='center',linewidth=2)
#df['ingredient-6'].value_counts().sort_values(ascending=False).head(30).plot.bar(width=0.5,edgecolor='c',align='center',linewidth=2)

plt.xlabel('Ingredients',fontsize=20)
plt.ylabel('Number',fontsize=20)
ax.tick_params(labelsize=20)
plt.title('50 Most common ingredients',fontsize=25)
plt.grid()
plt.ioff()

i1 = df['ingredient-1'].value_counts().sort_values(ascending=False).head(30)
i2 = df['ingredient-2'].value_counts().sort_values(ascending=False).head(30)
i3 = df['ingredient-3'].value_counts().sort_values(ascending=False).head(30)
i4 = df['ingredient-4'].value_counts().sort_values(ascending=False).head(30)
i5 = df['ingredient-5'].value_counts().sort_values(ascending=False).head(30)
i6 = df['ingredient-6'].value_counts().sort_values(ascending=False).head(30)
names1 = list(i1.keys())
names2 = list(i2.keys())
names3 = list(i3.keys())
names4 = list(i4.keys())
names5 = list(i5.keys())
names6 = list(i6.keys())
print(names1)
values1 = list(i1)
values2 = list(i2)
values3 = list(i3)
values4 = list(i4)
values5 = list(i5)
values6 = list(i6)
print(values1)

fig, axs = plt.subplots(figsize=(16,7))
axs.scatter(names1, values1)
axs.scatter(names2, values2)
axs.scatter(names3, values3)
axs.scatter(names4, values4)
axs.scatter(names5, values5)
axs.scatter(names6, values6)

