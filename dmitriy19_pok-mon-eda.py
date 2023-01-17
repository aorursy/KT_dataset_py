import pandas as pd

import seaborn as sns

import numpy as np



from collections import Counter



import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/Pokemon.csv', index_col=0)
data.head()
data.info()
data[['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].describe()
sns.set(style="whitegrid")

sns.set_color_codes("pastel")
colors = ["#a040a0", "#a890f0", "#7038f8", "#e0c068", "#e898e8",

          "#78c850", "#c03028", "#f85888", "#b8b8d0", "#98d8d8", 

          "#b8a038", "#705848", "#6890f0", "#f8d030", "#f08030",

          "#705898", "#a8b820", "#8a8a59"]
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

with sns.color_palette(colors, n_colors=18, desat=.9):

    sns.countplot('Type 1', data=data, ax=ax1)

    sns.countplot('Type 2', data=data, ax=ax2)
plot = sns.pairplot(data[['Total', 'HP', 'Attack', 'Defense', 'Speed']])
data['Generation'].unique()
Counter(data['Legendary'])
plot = sns.swarmplot(x="Generation", y="Total", hue="Legendary", data=data)
f, ax = plt.subplots(figsize=(14, 6))

with sns.color_palette(colors, n_colors=18, desat=.9):

    sns.boxplot(data['Total'], groupby=data['Type 1'])
ax = sns.kdeplot(data['Attack'], data['Defense'],

             cmap="Reds", shade=True, shade_lowest=False)