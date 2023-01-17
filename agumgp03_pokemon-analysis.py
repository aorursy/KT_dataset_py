import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/pokemon/Pokemon.csv')
df.head()
df.columns

print(df['Name'][0:3])

print(df[['Name','Type 1','Type 2']])



print(df.iloc[0:3])

for index, row in df.iterrows():

    print(index, row)

    df.loc[df['Type 1'] == 'Fire']



df.iloc[3,2]
df.sort_values(['Name'], ascending=False)

df['Name'][0:5].sort_values()
df['rank_total'] = (df['HP'] + df['Attack'] + df['Defense'] + df['Sp. Atk'] + df['Sp. Def'] + df['Speed']) / 8

df.sort_values(['rank_total'], ascending=False)
df = df.drop(columns='rank_total')

df
new_df = df.loc[(df['Type 1'] == 'Fire') & (df['HP'] > 70)]
new_df = new_df.sort_values(['HP'], ascending=False)

new_df.reset_index(drop=True, inplace=True)

new_df
new_new = new_df.groupby(['Type 2']).mean().sort_values('HP', ascending=False)

new_new

new_new.columns

highest_list = new_new.iloc[:,2]

highest_hp = highest_list.values.tolist()

print(highest_hp)

import matplotlib.pyplot as plt

import numpy as np
highest_hp
label = ['Steel', 'Psychic', 'Fighting', 'Flying', 'Normal', 'Water', 'Dragon']

highest_hp



bars = plt.bar(label,highest_hp)

plt.title('Pokemon HIghest HP')

plt.ylabel('HP')

plt.show()