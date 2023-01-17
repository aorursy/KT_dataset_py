# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import seaborn as sns
raw = pd.read_csv('/kaggle/input/pokemon/Pokemon.csv')
df = raw.copy()

df.head()
df.Generation.unique()
pd.crosstab(df['Generation'], columns='count', colnames=[''])
df[df.Generation == 1].head(10)
df.groupby('Generation')['Legendary'].sum()
df[(df.Generation == 3) & (df.Legendary == True)]
plt.figure(figsize=(20,8))

crosstab = pd.crosstab(df['Generation'], df['Type 1'])

sns.heatmap(crosstab, annot=True, cmap=sns.color_palette("Purples"), cbar=False, linewidths=.5)

plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)

plt.show()
plt.figure(figsize=(20,8))

crosstab = pd.crosstab(df['Generation'], df['Type 1']).apply(lambda x: round(x/x.sum(),3), axis=1)

sns.heatmap(crosstab, annot=True, cmap=sns.color_palette("Purples"), cbar=False, linewidths=.5)

plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)

plt.show()
plt.figure(figsize=(20,8))

crosstab = pd.crosstab(df['Generation'], df['Type 2']).apply(lambda x: round(x/x.sum(),3), axis=1)

sns.heatmap(crosstab, annot=True, cmap=sns.color_palette("Purples"), cbar=False, linewidths=.5)

plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)

plt.show()
plt.figure(figsize=(20,8))

crosstab = pd.crosstab(df['Type 1'], df['Type 2'])

sns.heatmap(crosstab, annot=True, cmap=sns.color_palette("Purples"), cbar=False, linewidths=.5)

plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)

plt.show()
pd.melt(df, id_vars=['Type 1', 'Type 2'], value_vars=['HP']).groupby(['Type 1', 'Type 2']).mean().sort_values(by='value', ascending=False)
pd.melt(df, id_vars=['Type 1', 'Type 2'], value_vars=['Attack']).groupby(['Type 1', 'Type 2']).mean().sort_values(by='value', ascending=False)
pd.melt(df, id_vars=['Type 1', 'Type 2'], value_vars=['Defense']).groupby(['Type 1', 'Type 2']).mean().sort_values(by='value', ascending=False)
pd.melt(df, id_vars=['Type 1', 'Type 2'], value_vars=['Sp. Atk']).groupby(['Type 1', 'Type 2']).mean().sort_values(by='value', ascending=False)
pd.melt(df, id_vars=['Type 1', 'Type 2'], value_vars=['Sp. Def']).groupby(['Type 1', 'Type 2']).mean().sort_values(by='value', ascending=False)
pd.melt(df, id_vars=['Type 1', 'Type 2'], value_vars=['Speed']).groupby(['Type 1', 'Type 2']).mean().sort_values(by='value', ascending=False)
pd.melt(df, id_vars=['Type 1', 'Type 2'], value_vars=['Total']).groupby(['Type 1', 'Type 2']).mean().sort_values(by='value', ascending=False)
df.groupby(['Type 1', 'Type 2'])[['Attack', 'Defense']].mean().sort_values(['Attack', 'Defense'], ascending=False)
df.groupby(['Type 1', 'Type 2'])[['Sp. Atk', 'Sp. Def']].mean().sort_values(['Sp. Atk', 'Sp. Def'], ascending=False)
df.groupby(['Type 1', 'Type 2'])[['Attack', 'Sp. Def']].mean().sort_values(['Attack', 'Sp. Def'], ascending=False)
df.groupby(['Type 1', 'Type 2'])[['Sp. Atk', 'Defense']].mean().sort_values(['Sp. Atk', 'Defense'], ascending=False)
def plot_pokemon_generation_stats(stat):

    fig, ax = plt.subplots(6,1,figsize=(10,12), constrained_layout=True)

    sns.kdeplot(df[df.Generation == 1][stat], color='#F3370C', shade=True, ax=ax[0])

    sns.kdeplot(df[df.Generation == 2][stat], color='#0C87F2', shade=True, ax=ax[1])

    sns.kdeplot(df[df.Generation == 3][stat], color='#F8CD0D',shade=True, ax=ax[2])

    sns.kdeplot(df[df.Generation == 4][stat], color='#59D10A',shade=True, ax=ax[3])

    sns.kdeplot(df[df.Generation == 5][stat], color='#BB0BE2',shade=True, ax=ax[4])

    sns.kdeplot(df[df.Generation == 6][stat], color='#0AD1D1',shade=True, ax=ax[5])



    ylim = max([ax[0].get_ylim()[1], ax[1].get_ylim()[1], ax[2].get_ylim()[1], ax[3].get_ylim()[1], ax[4].get_ylim()[1], ax[5].get_ylim()[1]])

    xmax = max([ax[0].get_xlim()[1], ax[1].get_xlim()[1], ax[2].get_xlim()[1], ax[3].get_xlim()[1], ax[4].get_xlim()[1], ax[5].get_xlim()[1]])



    for i in range(0,6):

        ax[i].grid(False)

        ax[i].set_title(f'Generation {i+1}')

        ax[i].set_ylim(0, ylim)

        ax[i].set_xlim(-25, xmax)



    fig.suptitle(f'Pokemon {stat.capitalize()} Distribution for Each Generation', fontsize=24)

    fig.text(-0.02, 0.5, "Density", ha="center", va="center", fontsize=22, rotation=90)

    fig.text(0.5,0-.02, f"{stat.capitalize()}", ha="center", va="center", fontsize=22)



    plt.show()
plot_pokemon_generation_stats('HP')
plot_pokemon_generation_stats('Attack')
plot_pokemon_generation_stats('Defense')
plot_pokemon_generation_stats('Sp. Atk')
plot_pokemon_generation_stats('Sp. Def')
plot_pokemon_generation_stats('Speed')
plot_pokemon_generation_stats('Total')