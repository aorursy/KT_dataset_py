# Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, gc, warnings

warnings.filterwarnings("ignore")



import plotly.express as px

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set_color_codes("muted")
# Load Data

url = '../input/income-and-happiness-correction/happyscore_income.csv'

df = pd.read_csv(url, header='infer')



# Dropping Columns

drp_cols = ['region','country.1','adjusted_satisfaction','std_satisfaction','median_income']

df.drop(drp_cols,axis=1,inplace=True)



# Inspect

df.head()
# Stat Summary

df.describe().T
# Correlation Heatmap



corr = df[df.columns[1:]].corr()



plt.figure(figsize=(10,8))



ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=False, annot=True,fmt='.1f')



ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

)



plt.title("Correlation Heatmap", fontsize=20)

plt.show()

'''Data Visualisation - Avg. Satisfaction vs Avg. Income'''



# Plot Config

fig, axes = plt.subplots(1, 2, figsize=(15,10))

plt.subplots_adjust(wspace=0.5)

fig.suptitle('Satisfaction vs Income', fontsize=20)

axes = axes.flatten()



temp = df[['country','avg_satisfaction','avg_income']]



# Highest Satisfaction Score

tempL = temp.nlargest(3, columns=['avg_satisfaction'])

tempL.reset_index(drop=True, inplace=True)



# Lowest Satisfaction Score

tempS = temp.nsmallest(3, columns=['avg_satisfaction'])

tempS.reset_index(drop=True, inplace=True)



# Plot

sns.barplot(ax=axes[0], data=tempL, x='avg_satisfaction', y='avg_income', hue="country", palette = 'magma')          

axes[0].set_title("Top 3 Countries with High Satisfaction Score")



sns.barplot(ax=axes[1], data=tempS, x='avg_satisfaction', y='avg_income', hue="country", palette = 'viridis')          

axes[1].set_title("Top 3 Countries with Low Satisfaction Score") 

# Plot Config

fig, axes = plt.subplots(1, 2, figsize=(15,10))

plt.subplots_adjust(wspace=0.5)

fig.suptitle('Satisfaction vs Income Inequality', fontsize=20)

axes = axes.flatten()



temp = df[['country','avg_satisfaction','income_inequality']]



# Highest Satisfaction Score

tempL = temp.nlargest(3, columns=['avg_satisfaction'])

tempL.reset_index(drop=True, inplace=True)



# Lowest Satisfaction Score

tempS = temp.nsmallest(3, columns=['avg_satisfaction'])

tempS.reset_index(drop=True, inplace=True)



# Plot

sns.barplot(ax=axes[0], data=tempL, x='avg_satisfaction', y='income_inequality', hue="country", palette = 'magma')          

axes[0].set_title("Top 3 Countries with High Satisfaction Score")



sns.barplot(ax=axes[1], data=tempS, x='avg_satisfaction', y='income_inequality', hue="country", palette = 'viridis')          

axes[1].set_title("Top 3 Countries with Low Satisfaction Score") 

# Plot Config

fig, axes = plt.subplots(1, 2, figsize=(15,10))

plt.subplots_adjust(wspace=0.5)

fig.suptitle('Satisfaction vs GDP', fontsize=20)

axes = axes.flatten()



temp = df[['country','avg_satisfaction','GDP']]



# Highest Satisfaction Score

tempL = temp.nlargest(3, columns=['avg_satisfaction'])

tempL.reset_index(drop=True, inplace=True)



# Lowest Satisfaction Score

tempS = temp.nsmallest(3, columns=['avg_satisfaction'])

tempS.reset_index(drop=True, inplace=True)



# Plot

sns.barplot(ax=axes[0], data=tempL, x='avg_satisfaction', y='GDP', hue="country", palette = 'magma')          

axes[0].set_title("Top 3 Countries with High Satisfaction Score")



sns.barplot(ax=axes[1], data=tempS, x='avg_satisfaction', y='GDP', hue="country", palette = 'viridis')          

axes[1].set_title("Top 3 Countries with Low Satisfaction Score") 
# Plot Config

fig, axes = plt.subplots(1, 2, figsize=(15,10))

plt.subplots_adjust(wspace=0.5)

fig.suptitle('Happiness vs Satisfaction', fontsize=20)

axes = axes.flatten()



temp = df[['country','avg_satisfaction','happyScore']]



# Highest Happiness Score

tempL = temp.nlargest(3, columns=['happyScore'])

tempL.reset_index(drop=True, inplace=True)



# Lowest Happiness Score

tempS = temp.nsmallest(3, columns=['happyScore'])

tempS.reset_index(drop=True, inplace=True)



# Plot

sns.barplot(ax=axes[0], data=tempL, x='avg_satisfaction', y='happyScore', hue="country", palette = 'magma')          

axes[0].set_title("Top 3 Countries with High Happiness Score")



sns.barplot(ax=axes[1], data=tempS, x='avg_satisfaction', y='happyScore', hue="country", palette = 'viridis')          

axes[1].set_title("Top 3 Countries with Low Happiness Score") 
# Plot Config

fig, axes = plt.subplots(1, 2, figsize=(15,10))

plt.subplots_adjust(wspace=0.5)

fig.suptitle('Happiness vs Avg. Income', fontsize=20)

axes = axes.flatten()



temp = df[['country','avg_income','happyScore']]



# Highest Happiness Score

tempL = temp.nlargest(3, columns=['happyScore'])

tempL.reset_index(drop=True, inplace=True)



# Lowest Happiness Score

tempS = temp.nsmallest(3, columns=['happyScore'])

tempS.reset_index(drop=True, inplace=True)



# Plot

sns.barplot(ax=axes[0], data=tempL, x='happyScore', y='avg_income', hue="country", palette = 'magma')          

axes[0].set_title("Top 3 Countries with High Happiness Score")



sns.barplot(ax=axes[1], data=tempS, x='happyScore', y='avg_income', hue="country", palette = 'viridis')          

axes[1].set_title("Top 3 Countries with Low Happiness Score") 
# Plot Config

fig, axes = plt.subplots(1, 2, figsize=(15,10))

plt.subplots_adjust(wspace=0.5)

fig.suptitle('Happiness vs Income Inequality', fontsize=20)

axes = axes.flatten()



temp = df[['country','income_inequality','happyScore']]



# Highest Happiness Score

tempL = temp.nlargest(3, columns=['happyScore'])

tempL.reset_index(drop=True, inplace=True)



# Lowest Happiness Score

tempS = temp.nsmallest(3, columns=['happyScore'])

tempS.reset_index(drop=True, inplace=True)



# Plot

sns.barplot(ax=axes[0], data=tempL, x='happyScore', y='income_inequality', hue="country", palette = 'magma')          

axes[0].set_title("Top 3 Countries with High Happiness Score")



sns.barplot(ax=axes[1], data=tempS, x='happyScore', y='income_inequality', hue="country", palette = 'viridis')          

axes[1].set_title("Top 3 Countries with Low Happiness Score") 
# Plot Config

fig, axes = plt.subplots(1, 2, figsize=(15,10))

plt.subplots_adjust(wspace=0.5)

fig.suptitle('Happiness vs GDP', fontsize=20)

axes = axes.flatten()



temp = df[['country','GDP','happyScore']]



# Highest Happiness Score

tempL = temp.nlargest(3, columns=['happyScore'])

tempL.reset_index(drop=True, inplace=True)



# Lowest Happiness Score

tempS = temp.nsmallest(3, columns=['happyScore'])

tempS.reset_index(drop=True, inplace=True)



# Plot

sns.barplot(ax=axes[0], data=tempL, x='happyScore', y='GDP', hue="country", palette = 'magma')          

axes[0].set_title("Top 3 Countries with High Happiness Score")



sns.barplot(ax=axes[1], data=tempS, x='happyScore', y='GDP', hue="country", palette = 'viridis')          

axes[1].set_title("Top 3 Countries with Low Happiness Score") 