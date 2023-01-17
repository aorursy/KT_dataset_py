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
pokemon_df  = pd.read_csv('../input/pokemon/pokemon.csv')
pokemon_df.head()
import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline
sns.swarmplot(x=pokemon_df['is_legendary'], y=pokemon_df['sp_attack'])

plt.savefig('sp_attack vs is_legendary swarm.png')
ohe_pokemon_df = pd.get_dummies(pokemon_df)

ohe_pokemon_df.head()
ohe_pokemon_df.columns
pokemon_df.columns
sns.lmplot(x='sp_attack', y='sp_defense', hue='is_legendary', data=pokemon_df)

plt.savefig('lmplot sp_attack vs sp_defense.png')
sns.lineplot(x=pokemon_df['sp_attack'], y=pokemon_df['sp_defense'], hue=pokemon_df['is_legendary'])

plt.savefig('lineplot sp_attack vs sp_defense.png')
sns.scatterplot(x=pokemon_df['sp_attack'], y=pokemon_df['sp_defense'], hue=pokemon_df['is_legendary'])

plt.savefig('scatter plot sp_attack vs sp_defense.png')
sns.regplot(x=pokemon_df['sp_attack'], y=pokemon_df['sp_defense'])

plt.savefig('regplot sp_attack vs sp_defense.png')
sns.distplot(a=pokemon_df['sp_attack'], kde=False)

plt.savefig('histogram sp_attack.png')
sns.kdeplot(data=pokemon_df['sp_attack'], shade=True)

plt.savefig('kde sp_attack.png')
sns.kdeplot(data=pokemon_df['sp_defense'], shade=True)

plt.savefig('kde sp_defense.png')
sns.jointplot(x=pokemon_df['sp_attack'], y=pokemon_df['sp_defense'], kind='kde')

plt.savefig('joint plot sp_Attack vs sp_defense.png')
sns.swarmplot(x=pokemon_df['is_legendary'], y=pokemon_df['sp_defense'])

plt.savefig('swarm plot is_legendary vs sp_defense.png')