import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
# from beakerx import *
warnings.filterwarnings('ignore')
%matplotlib inline
color = sns.color_palette()
data = pd.read_csv('../input/pokemon.csv')
data.head()
data.columns
data.shape
data.isnull().sum()
data.nunique()
# plt.figure(figsize=(14,6))
sns.set_style('whitegrid')
sns.lmplot(
    x="Attack",
    y="Defense",
    data=data,
    fit_reg=False,
    hue='Legendary',
    palette="Set1")
sns.set_style('darkgrid')  #changes the background of the plot
plt.figure(figsize=(14, 6))
sns.regplot(
    x="Attack", y="Defense", data=data,
    fit_reg=True)  #fit_Reg fits a regression line
plt.figure(figsize=(20, 6))
sns.set_style('whitegrid')
sns.lmplot(
    x="Attack",
    y="Defense",
    data=data,
    fit_reg=False,
    hue='Legendary',
    col="Generation",
    aspect=0.4,
    size=10)
plt.figure(figsize=(14, 6))
sns.set_style('whitegrid')
sns.regplot(x="Legendary", y="Speed", data=data)
plt.figure(figsize=(14, 6))
sns.set_style("ticks")
sns.regplot(x="Legendary", y="Speed", data=data, x_jitter=0.3)
plt.figure(figsize=(14, 6))
sns.set_style("ticks")
sns.regplot(x="Attack", y="Legendary", data=data, logistic=True)
plt.figure(figsize=(12, 6))
ax = sns.distplot(data['Attack'], kde=False)
ax.set_title('Attack')
plt.figure(figsize=(12, 6))
ax = sns.distplot(
    data['Defense'], kde=True,
    norm_hist=False)  #norm_hist normalizes the count
ax.set_title('Defense')
plt.show()
plt.figure(figsize=(12, 6))
ax = sns.distplot(data['Speed'], rug=True)
ax.set_title('Speed')
plt.show()
plt.figure(figsize=(12, 6))
ax = sns.kdeplot(data['HP'], shade=True, color='g')
ax.set_title('HP')
plt.show()
plt.figure(figsize=(12, 6))
sns.stripplot(
    y='HP', data=data, jitter=0.1,
    color='g')  #jitter option to spread the points
plt.figure(figsize=(12, 6))
sns.boxplot(y='Speed', data=data, width=.6)
plt.figure(figsize=(12, 6))
sns.jointplot(x='HP', y='Speed', data=data)
plt.figure(figsize=(12, 6))
sns.jointplot(x='HP', y='Speed', data=data, kind='kde')
plt.figure(figsize=(12, 6))
sns.jointplot(x='HP', y='Speed', data=data, kind='hex')
sns.pairplot(data)
sns.pairplot(data, hue='Legendary')
sns.pairplot(
    data,
    hue='Legendary',
    vars=['Speed', 'HP', 'Attack', 'Defense', 'Generation'],
    diag_kind='kde')
plt.figure(figsize=(20, 6))
ax = sns.countplot(x="Type 1", data=data, color='c')
plt.figure(figsize=(20, 6))
sns.countplot(
    x="Type 1", data=data, hue='Legendary',
    dodge=False)  #dodge = False option is used to make stacked plots
sns.set_style('darkgrid')
plt.figure(figsize=(20, 6))
sns.barplot(x="Type 1", y='Speed', data=data, color='c')
sns.set_style('darkgrid')
plt.figure(figsize=(20, 6))
sns.barplot(x="Type 1", y='Speed', data=data, hue='Legendary')
plt.figure(figsize=(20, 6))
sns.pointplot(x="Generation", y='Speed', data=data, hue='Legendary')
plt.figure(figsize=(12, 6))
sns.stripplot(x="Generation", y="Speed", data=data)
plt.figure(figsize=(12, 6))
sns.stripplot(x="Generation", y="Speed", data=data, jitter=0.3)
sns.set_style('ticks')
plt.figure(figsize=(12, 6))
sns.swarmplot(x="Generation", y="Speed", data=data, hue='Legendary')
plt.figure(figsize=(12, 6))
sns.boxplot(x="Generation", y="Speed", data=data, hue='Legendary')
plt.figure(figsize=(12, 6))
sns.violinplot(x="Generation", y="Speed", data=data, hue='Legendary')
plt.figure(figsize=(12, 6))
sns.violinplot(
    x="Generation", y="Speed", data=data, hue='Legendary', split=True)
plt.figure(figsize=(12, 6))
sns.violinplot(
    x="Generation",
    y="Speed",
    data=data,
    hue='Legendary',
    split=True,
    scale='count')
plt.figure(figsize=(12, 6))
sns.violinplot(
    x="Generation",
    y="Speed",
    data=data,
    hue='Legendary',
    split=True,
    inner='quartile')
plt.figure(figsize=(12, 6))
sns.violinplot(
    x="Generation",
    y="Speed",
    data=data,
    hue='Legendary',
    split=True,
    inner='stick') #show each datapoint
plt.figure(figsize=(12, 6))
sns.violinplot(
    x="Generation",
    y="Speed",
    data=data,
    hue='Legendary',
    split=True,
    inner='stick',
    bw=.2)
g = sns.PairGrid(
    data,
    x_vars=["Generation", "Legendary"],
    y_vars=["Speed", "HP", "Attack"],
    aspect=.75,
    size=8)
g.map(sns.violinplot, palette="pastel")
g = sns.FacetGrid(data=data, col='Generation', col_wrap=3)

g.map(plt.hist, "Speed")
g = sns.FacetGrid(data=data, col='Generation', col_wrap=3, hue="Legendary")

g.map(sns.regplot, "Speed", "HP", fit_reg=False).add_legend()

g = sns.FacetGrid(
    data=data, col='Generation', row='Legendary', margin_titles=True)

g.map(sns.regplot, "Speed", "HP", fit_reg=False)
g = sns.FacetGrid(
    data=data, col='Generation', margin_titles=True, size=4, aspect=.8)

g.map(sns.distplot, "Speed")
g = sns.FacetGrid(
    data=data,
    col='Generation',
    margin_titles=True,
    size=4,
    aspect=.8,
    hue='Legendary')

g.map(sns.violinplot, "Speed")
dragon = data.loc[data['Type 1']=='Dragon']
g = sns.PairGrid(
    dragon,
    vars=["Attack", "Defense"],
    size=5,
)
g.map(plt.scatter, s=4 * dragon.HP, alpha=.5)
