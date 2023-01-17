#Import libraries

import numpy as np 

import pandas as pd 

import seaborn as sns

from matplotlib import pyplot as plt

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("../input/World_Champs_Men's_All-Around.csv")

df.head(12)
pd.DataFrame(df.groupby(['Nationality'])['Name'].count()/6).astype(int)
sns.set_style('darkgrid', {'axis.facecolor':'black'})

vis1 = sns.boxplot(data=df, y='Name', x='Rank')
sns.set_style('darkgrid', {'axis.facecolor':'black'})

f, axes= plt.subplots(1, 3, figsize=(15,4))

ax = sns.boxplot(x='Name', y='Diff', data=df,ax = axes[0])

ax = ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

ax = sns.boxplot(x='Name', y='Exec', data=df,ax = axes[1])

ax = ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

ax = sns.boxplot(x='Name', y='Total', data=df,ax = axes[2])

ax = ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
sns.set_style('darkgrid', {'axis.facecolor':'black'})

g = sns.FacetGrid(df, row="Apparatus", size=6)

g = g.map(sns.barplot, "Total", "Name",  palette="Blues_d")

g = g.map(sns.barplot, "Diff", "Name",  palette="Reds")

g = g.axes[5,0].set_xlabel('Toal score and Diff score')
sns.set_style('dark', {'axis.facecolor':'black'})

g = sns.FacetGrid(df, row="Apparatus", size=6)

kws= dict(s=6, linewidth=0.5, edgecolor='black')

g = g.map(sns.stripplot, "Rank", 'Name', color='Red', **kws)

g = g.map(sns.stripplot, "Overall Rank", 'Name',color='blue', **kws)

g = g.axes[5,0].set_xlabel('Overall Rank(Blue) and Rank(Red)')
sns.set_style('darkgrid', {'axis.facecolor':'black'})

vis1 = sns.boxplot(data=df, y='Nationality', x='Rank')
sns.set_style('darkgrid', {'axis.facecolor':'black'})

f, axes= plt.subplots(1, 3, figsize=(15,4))

ax = sns.boxplot(x='Nationality', y='Diff', data=df,ax = axes[0])

ax = ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

ax = sns.boxplot(x='Nationality', y='Exec', data=df,ax = axes[1])

ax = ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

ax = sns.boxplot(x='Nationality', y='Total', data=df,ax = axes[2])

ax = ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
sns.set_style('darkgrid', {'axis.facecolor':'black'})

g = sns.FacetGrid(df, row="Apparatus", size=6)

g = g.map(sns.barplot, "Total", "Nationality",  palette="Blues_d")

g = g.map(sns.barplot, "Diff", "Nationality",  palette="Reds")

g = g.axes[5,0].set_xlabel('Toal score and Diff score')