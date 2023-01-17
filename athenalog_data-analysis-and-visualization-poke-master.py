# import all libs

from os import path

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

%matplotlib inline
# import dataset

df = pd.read_csv('../input/Pokemon.csv')
df.head()
#create a grid scatter plot to view relationship betwenn all parameters

df_cols = df[['Type 1','HP','Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]



with sns.color_palette(["#8ED752", "#F95643", "#53AFFE", "#C3D221", "#BBBDAF",

    "#AD5CA2", "#F8E64E", "#F0CA42", "#F9AEFE", "#A35449",

    "#FB61B4", "#CDBD72", "#7673DA", "#66EBFF", "#8B76FF",

    "#8E6856", "#C3C1D7", "#75A4F9"], n_colors=18, desat=.9):

    g=sns.PairGrid(df_cols,hue='Type 1')

    g = g.map_offdiag(plt.scatter)

    g = g.map_diag(plt.hist)

    g.add_legend()
#view the nuumber of pokemons for Type 1 and Type 2 using one plot

f, (ax1,ax2) = plt.subplots(2,1,figsize=(15, 8),sharex=True)



sns.countplot('Type 1',data=df,ax=ax1)

sns.countplot('Type 2',data=df,ax=ax2)
#create a pie chart to show number of pokemons of each type(Type 1)

#get the name of each type(Type 1). These will be the labels in pie chart

types = df['Type 1'].unique()





#add index column

df['index'] = df.index

df.head()
# Get the number of pokemons for each type (Type 1)

number_for_each_type = df.groupby('Type 1').index.nunique()



# covert it into an array

number_for_each_type = np.array(number_for_each_type)

number_for_each_type
#create explode parameter for the pie chart

explode = np.zeros_like(types,dtype=float)

explode[-1]=0.1

explode





#create colors for the pie chart

colors = ["#8ED752", "#F95643", "#53AFFE", "#C3D221", "#BBBDAF",

    "#AD5CA2", "#F8E64E", "#F0CA42", "#F9AEFE", "#A35449",

    "#FB61B4", "#CDBD72", "#7673DA", "#66EBFF", "#8B76FF",

    "#8E6856", "#C3C1D7", "#75A4F9"]
#create pie chart

plt.figure(figsize=(9,9))

plt.pie(number_for_each_type, labels=types,explode=explode,shadow=True,colors=colors,autopct='%1.1f%%')