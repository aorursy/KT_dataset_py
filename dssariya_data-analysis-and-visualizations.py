import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from os import path

import numpy as np

%matplotlib inline
mydf_pkmn = pd.read_csv('../input/Pokemon.csv')
#Take a peek at your raw data

peek = mydf_pkmn.head(20)

print(peek)
#Review the dimensions of your dataset.

shape = mydf_pkmn.shape 

print(shape)
#Review the data types of attributes in your data



types = mydf_pkmn.dtypes 

print(types)
# Summarize your data using descriptive statistics.



pd.set_option('display.width', 100) 

pd.set_option('precision', 3) 

description = mydf_pkmn.describe() 

print(description)
# Understand the relationships in your data using correlations.

pd.set_option('display.width', 100)

pd.set_option('precision', 3)

correlations = mydf_pkmn.corr(method='pearson') 

print(correlations)
# Review the skew of the distributions of each attribute.

skew = mydf_pkmn.skew() 

print(skew)
# unimodel

# histograms

mydf_pkmn.hist()

plt.show()
# box and whisker plots



mydf_pkmn.plot(kind='box', subplots=True) 

plt.show()
sns.boxplot(y="HP", data=mydf_pkmn);
sns.boxplot(data=mydf_pkmn);
#create a grid scatter plot to view relationship betwenn all parameters

df_cols = mydf_pkmn[['Type 1','HP','Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]



with sns.color_palette(["#8ED752", "#F95643", "#53AFFE", "#C3D221", "#BBBDAF",

    "#AD5CA2", "#F8E64E", "#F0CA42", "#F9AEFE", "#A35449",

    "#FB61B4", "#CDBD72", "#7673DA", "#66EBFF", "#8B76FF",

    "#8E6856", "#C3C1D7", "#75A4F9"], n_colors=18, desat=.9):

    g=sns.PairGrid(df_cols,hue='Type 1')

    g = g.map_offdiag(plt.scatter)

    g = g.map_diag(plt.hist)

    g.add_legend()