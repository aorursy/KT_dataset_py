# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/fifa-18-demo-player-dataset/CompleteDataset.csv", index_col = 0)
df.head()
import copy, re

import numpy as np



footballers = df.copy()

footballers['Unit'] = footballers['Value'].str[-1]

footballers['Value (M)'] = np.where(footballers['Unit'] == '0', 0 , footballers['Value'].str[1:-1].replace(r'[a-zA-Z]', ''))

footballers['Value (M)'] = footballers['Value (M)'].astype(float)

footballers['Value (M)'] = np.where(footballers['Unit'] == 'M', footballers['Value (M)'], footballers['Value (M)']/1000)



footballers = footballers.assign(Value =footballers['Value (M)'], Position = footballers['Preferred Positions'].str.split().str[0] )

footballers.head()
import seaborn as sns
## FacetGrid

df = footballers[footballers['Position'].isin(['ST', 'GK'])]

g = sns.FacetGrid(df, col="Position")
df = footballers[footballers['Position'].isin(['ST', 'GK'])]

g = sns.FacetGrid(df, col="Position")

g.map(sns.kdeplot, "Overall")
g = sns.FacetGrid(footballers, col="Position", col_wrap=6 )

g.map(sns.kdeplot, "Overall")
df = footballers[footballers['Position'].isin(['ST', 'GK'])]

df = df[df['Club'].isin(['Real Madrid CF', 'FC Barcelona', 'Atlético Madrid'])]

g = sns.FacetGrid(df, row='Position', col='Club')

g.map(sns.violinplot, "Overall")
## specify the order

df = footballers[footballers['Position'].isin(['ST', 'GK'])]

df = df[df['Club'].isin(['Real Madrid CF', 'FC Barcelona', 'Atlético Madrid'])]

g = sns.FacetGrid(df, row='Position', col='Club', row_order=['GK', 'ST'], col_order=['Atlético Madrid', 'FC Barcelona', 'Real Madrid CF'])

g.map(sns.violinplot, "Overall")

sns.pairplot(footballers[['Overall', 'Potential', 'Value']])
##Multivariate scatter plots

sns.lmplot(x="Value", y='Overall', hue='Position', data=footballers.loc[footballers['Position'].isin(['ST', 'RW', 'LW'])], fit_reg=False)

sns.lmplot(x='Value', y='Overall', hue='Position', markers=['o', 'x', '*'], 

           data=footballers.loc[footballers['Position'].isin(['ST', 'RW', 'LW'])],fit_reg=False)
f = (footballers

         .loc[footballers['Position'].isin(['ST', 'GK'])]

         .loc[:, ['Value', 'Overall', 'Aggression', 'Position']]

    )

f = f[f["Overall"] >= 80]

f = f[f["Overall"] < 85]

f['Aggression'] = f['Aggression'].astype(float)



sns.boxplot(x="Overall", y='Aggression', hue='Position', data=f)
#heatmap

f = (

    footballers.loc[:, ['Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control']]

        .applymap(lambda v: int(v) if str.isdecimal(v) else np.nan)

        .dropna()

).corr()



sns.heatmap(f, annot=True)
##Parallel Coordinates

from pandas.plotting import parallel_coordinates

f = (

    footballers.iloc[:, 12:17]

        .loc[footballers['Position'].isin(['ST', 'GK'])]

        .applymap(lambda v: int(v) if str.isdecimal(v) else np.nan)

        .dropna()

)

f['Position'] = footballers['Position']

f = f.sample(200)



parallel_coordinates(f, 'Position')
## Pokemon dataset

df = pd.read_csv("../input/pokemon/Pokemon.csv", index_col=0)

df.head()

#FacetGrid on Legendary column values

g = sns.FacetGrid(df, col='Legendary')

g.map(sns.kdeplot, 'Attack')

g = sns.FacetGrid(df, row='Generation', col='Legendary')

g.map(sns.kdeplot, 'Attack')
#pair plot

sns.pairplot(df[['HP', 'Attack', 'Defense']])
sns.lmplot(x="Attack", y="Defense", hue="Legendary", markers=['x', 'o'], data=df, fit_reg=False)
sns.boxplot(x='Generation', y='Total', hue='Legendary', data=df)
sns.heatmap(df[['HP', 'Attack', 'Sp. Atk', 'Defense', 'Sp. Def', 'Speed' ]].corr(), annot=True)
df = df[['Attack', 'Sp. Atk', 'Defense', 'Sp. Def', 'Type 1']]

df = df[df['Type 1'].isin(["Psychic", "Fighting"])]

parallel_coordinates(df, 'Type 1')