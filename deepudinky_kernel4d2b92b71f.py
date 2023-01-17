import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv("../input/vgsales.csv")
df.head(5)
df.tail(5)
df.shape
df.dtypes
df.info()
df.describe()
sns.barplot(x='Genre',y='JP_Sales', data=df) 

plt.xticks(rotation=90)
y = df.groupby(['Year']).sum()

y = y['JP_Sales']

x = y.index.astype(int)

plt.figure(figsize=(10,6))

ax = sns.barplot(y = y, x = x)

ax.set_xlabel(xlabel='$ Millions', fontsize=14)

ax.set_xticklabels(labels = x, fontsize=10, rotation=50)

ax.set_ylabel(ylabel='Year', fontsize=14)

ax.set_title(label='Game Sales in $ Millions Per Year', fontsize=20)

plt.show();
#Scatter plot on JP_Sales

sns.scatterplot("Rank","JP_Sales",data=df)
#Categorical plot on JP_Sales

sns.catplot(x="Genre", y="JP_Sales",data=df)
# Sunset data by Genre

puzzledata = df[df['Genre'] == 'Puzzle']

fightdata = df[df['Genre'] == 'Fighting']

platformdata = df[df['Genre'] == 'Platform']



sns.jointplot("Year", "Global_Sales",data=puzzledata, color='blue')

sns.jointplot("Year", "Global_Sales",data=fightdata, color='red')

sns.jointplot("Year", "Global_Sales",data=platformdata, color='green')
#Pair Plot for Genre Distribution by Market

Pair = df.drop(['Rank', 'Global_Sales'], axis=1)

sns.pairplot(Pair,hue='Genre')
# Regplot



fig, axs = plt.subplots(ncols=5,figsize=(20,6))



sns.regplot(x='Year', y='Global_Sales', data=puzzledata, ax=axs[0])

axs[0].set_title('Puzzle', fontsize=15)



sns.regplot(x='Year', y='Global_Sales', data=platformdata, ax=axs[1])

axs[1].set_title('Platform', fontsize=15)



sns.regplot(x='Year',y='Global_Sales', data=fightdata, ax=axs[2])

axs[2].set_title('Fighting', fontsize=15)
#Line plot

sns.lineplot(x="Year", y="JP_Sales", ci='sd', estimator=None, data=df);

plt.xticks(rotation=90);
#Point plot

sns.pointplot(x="Year", y="JP_Sales", data=df, capsize=5)

plt.xticks(rotation=90);
#histogram

sns.distplot(df['JP_Sales'], hist=False, bins=3)
#pairplot

sns.pairplot(df)
table_count = pd.pivot_table(df,values=['JP_Sales'],index=['Year'],columns=['Genre'],aggfunc='count',margins=True)

plt.figure(figsize=(19,16))

sns.heatmap(table_count['JP_Sales'],linewidths=1,annot=True,fmt='1.0f',vmin=0)

plt.title('Count of games')