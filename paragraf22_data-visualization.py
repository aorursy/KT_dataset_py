import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('/kaggle/input/world-university-rankings/cwurData.csv')
df.head()
df.info()
df.describe()
plt.figure(figsize=(16,9))

sns.heatmap(df.corr(),annot=True,linewidth=0.5,fmt='.2f',cmap='BuPu')

plt.suptitle("Heatmap")
plt.figure(figsize=(16,15))

plt.subplot(3,1,1)

plt.scatter(df['world_rank'],df["publications"],alpha=0.25,cmap='BuPu',edgecolors='face')

plt.xlabel("publications")

plt.ylabel("world_rank")

plt.subplot(3,1,2)

plt.scatter(df['world_rank'],df["patents"],alpha=0.25,cmap='BuPu',edgecolors='face')

plt.xlabel("patents")

plt.ylabel("world_rank")

plt.subplot(3,1,3)

plt.scatter(df['world_rank'],df["broad_impact"],alpha=0.25,cmap='BuPu',edgecolors='face')

plt.xlabel("broad_impact")

plt.ylabel("world_rank")

plt.show()
from collections import Counter

country_list = list(df['country'])

sorted_d = sorted(Counter(country_list).items() , key=lambda x: x[1],reverse = True)

top_ten = sorted_d[:10]



df_top_ten = pd.DataFrame(top_ten)

df_top_ten.rename(columns={0: "country", 1: "num_univ"},inplace=True)
plt.figure(figsize=(12,8))

sns.barplot(x = df_top_ten["country"],y=df_top_ten["num_univ"] ,palette=sns.cubehelix_palette(len(df_top_ten['country'])))

plt.xticks(rotation=45)

plt.xlabel("Country")

plt.ylabel('Number of universities in ranking')

plt.title('Top ten countries in ranking')
f, ax = plt.subplots(figsize=(16,9))

sns.pointplot(x='world_rank',y='score',data=df[:25],alpha=0.8, color='g' )

plt.xticks(rotation=45)

plt.xlabel("world rank",fontsize=15)

plt.ylabel("score",fontsize=15)

plt.grid()
df_scatter = df[["quality_of_education","score","quality_of_faculty"]]
pd.plotting.scatter_matrix(df_scatter, figsize=(10, 10),diagonal='kde',marker = ".")

plt.show()
fig1, ax1 = plt.subplots(figsize=(12,8))

explode = np.zeros(len(df_top_ten))

explode[0] = 0.1

ax1.pie(df_top_ten['num_univ'], labels=df_top_ten['country'], autopct='%1.1f%%',shadow=True, startangle=90,explode=explode )

ax1.axis('equal') 

plt.title('Top ten countries in ranking')

plt.show()
sns.set(style="ticks")

sns.jointplot(df['publications'],df['patents'], kind="hex", color="#4CB391",ratio=5,space=0)
sns.set(style="ticks")

sns.jointplot(df['publications'],df['patents'], kind="kde", color="#4CB391",ratio=3,space=0)
fig1, ax1 = plt.subplots(figsize=(12,9))

green_diamond = dict(markerfacecolor='g', marker='D')

ax1.set_title('Distribution of score')

ax1.boxplot(df['score'], flierprops=green_diamond,whis = 9)

sns.set(style="whitegrid")

ax = plt.subplots(figsize=(12,9))

ax = sns.violinplot(y=df["score"],inner="quartile")

ax.set_title('Distribution of score')
cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)

ax = plt.subplots(figsize=(14,9))

ax = sns.scatterplot(x="citations", y="alumni_employment",hue="score",size='score',

                     palette=cmap, sizes=(20, 300),

                     data=df)