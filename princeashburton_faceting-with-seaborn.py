

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))


pd.set_option('max_columns',None)
df = pd.read_csv("../input/fifa-18-demo-player-dataset/CompleteDataset.csv", index_col=0)

import re
footballers = df.copy()
footballers['Unit'] = df['Value'].str[-1]
footballers['Value (M)'] = np.where(footballers['Unit'] == '0', 0,
                                   footballers['Value'].str[1:-1].replace(r'[a-zA-Z]',''))
footballers['Value (M)'] = footballers['Value (M)'].astype(float)
footballers['Value (M)'] = np.where(footballers['Unit'] =='M',
                                   footballers['Value (M)'],
                                   footballers['Value (M)']/1000)
footballers = footballers.assign(Value=footballers['Value (M)'],
                                Postion=footballers['Preferred Positions'].str.split().str[0])
footballers.head(100)
footballers.info()
footballers.rename(columns={'Postion': 'Position'}, inplace=True)
footballers.info()
import seaborn as sns
df = footballers[footballers['Position'].isin(['ST','GK'])]
g = sns.FacetGrid(df, col="Position")
g.map(sns.kdeplot, "Overall")

df = footballers

g = sns.FacetGrid(df, col="Position", col_wrap=6)
g.map(sns.kdeplot, "Overall")
df = footballers[footballers['Position'].isin(['ST','GK'])]
df = df[df['Club'].isin(['Real Madrid CF', 'FC Barcelona','Atlético Madrid'])]

g = sns.FacetGrid(df, row="Position", col="Club")
g.map(sns.violinplot, "Overall")
footballers['Club']
df = footballers[footballers['Position'].isin(['CB'])]
df = df[df['Club'].isin(['Arsenal', 'Tottenham Hotspur','Manchester City', 'Chelsea','Liverpool', 'Manchest United'])]

g = sns.FacetGrid(df, row="Position", col="Club")
g.map(sns.violinplot, "Overall")
df = footballers[footballers['Position'].isin(['ST','GK'])]
df = df[df['Club'].isin(['Real Madrid CF', 'FC Barcelona','Atlético Madrid'])]

g = sns.FacetGrid(df, row="Position", col="Club",
                 row_order=['GK','ST'],
                 col_order=['Atlético Madrid','FC Barcelona','Real Madrid CF' ])
g.map(sns.violinplot, "Overall")
sns.pairplot(footballers[['Overall','Potential','Value']])
pokemon = pd.read_csv("../input/pokemon/Pokemon.csv", index_col=0)

g = sns.FacetGrid(pokemon, row="Legendary")
g.map(sns.kdeplot,"Attack")
g = sns.FacetGrid(pokemon, col="Legendary", row="Generation")
g.map(sns.kdeplot,"Attack")
sns.pairplot(pokemon[['HP','Attack','Defense']])
