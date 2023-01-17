# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # matplotlib for additional customization
import seaborn as sns # Seaborn for plotting and styling
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# import the dataset
pokemon = pd.read_csv('../input/Pokemon.csv')
# check the dataframe
pokemon.head()
# dataframe information
pokemon.info()
# check the correlation number of other stats vs. total strength
pokemon.corr()[['Total']]
# Which stats of a pokemon is the most important / correlated to the total strength of a pokemon?
# Special Attack is the most important stats of a pokemon (.75 correlated to the total strength of a pokemon)
# We should dig into the special attack feature next
sns.heatmap(pokemon.corr()[['Total']], annot=True)
sns.set(rc={'figure.figsize':(11.7,8.27)})
# Distribution of the special attack stats among Pokemon overall?
# The special attack number of most of the Pokemon is around 60, the histogram is right-skewed, which means 
# there are more Pokemon has lower than 60 special attack number compared to higher than average special attack number 
# Next, I'll figure out which type of Pokemon has a higher special attack number, so they are tend to have a better total strength number
plt.hist(pokemon['Sp. Atk'], bins=20)
plt.xlabel('Sp. Atk')
plt.ylabel('Number of Pokemon')
# Which type of Pokemon has a higher special attack number?
# According to the chart, Dragon and Flying are the top 2 types of Pokemon in terms of the special attach number (average more than 100).
# However, there are more variance for Dragon compared to Flying Pokemon.
# Conclusion: choose Flying Pokemon is a better choice to get a stable higher special attack feature.
sns.violinplot(x='Type 1', y='Sp. Atk', data=pokemon)