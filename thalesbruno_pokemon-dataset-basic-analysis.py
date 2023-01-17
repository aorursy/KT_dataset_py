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
import seaborn as sns

from matplotlib import pyplot as plt

sns.set()



print('Modules loaded')
pokemon_filepath = '../input/pokemon/Pokemon.csv'

pokemon = pd.read_csv(pokemon_filepath)

pokemon.head()
fig, axes = plt.subplots(2, 3, figsize=(18, 10))



fig.suptitle('Pokemon Stats by Generation')



sns.boxplot(ax=axes[0, 0], data=pokemon, x='Generation', y='Attack')

sns.boxplot(ax=axes[0, 1], data=pokemon, x='Generation', y='Defense')

sns.boxplot(ax=axes[0, 2], data=pokemon, x='Generation', y='Speed')

sns.boxplot(ax=axes[1, 0], data=pokemon, x='Generation', y='Sp. Atk')

sns.boxplot(ax=axes[1, 1], data=pokemon, x='Generation', y='Sp. Def')

sns.boxplot(ax=axes[1, 2], data=pokemon, x='Generation', y='HP')