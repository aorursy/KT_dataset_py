import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

df=pd.read_csv('../input/chess/games.csv')

df.head()
# rank groupings---> find most common opening by rank grouping

# most common opening move 
plt.style.use('dark_background')

sns.distplot(df.turns,color='white')

plt.show()

plt.style.use('default')

sns.jointplot(x='white_rating',y='black_rating',data=df,kind='hex',color='black')

plt.style.use('default')

colors = ["white", "darkgrey","red"]

customPalette = sns.set_palette(sns.color_palette(colors))

sns.violinplot(x="winner", y="turns", data=df,palette=customPalette)

plt.style.use('grayscale')



sns.regplot(x='black_rating',y='turns',data=df,scatter_kws={'s':2})
# rated games vs non rated games length

plt.style.use('default')

sns.countplot(df.rated)
plt.style.use('default')

sns.countplot(df.victory_status,order=df.victory_status.value_counts().iloc[:4].index)
my_colors=['red','blue']

df.groupby(['rated']).mean().turns.plot(kind='bar',color=my_colors)
sns.countplot(y="black_id", data=df, palette="Reds",

              order=df.black_id.value_counts().iloc[:10].index)
plt.style.use('default')

plt.figure(figsize=(30, 10))

sns.countplot(df.opening_name,order=df.opening_name.value_counts().iloc[:8].index)