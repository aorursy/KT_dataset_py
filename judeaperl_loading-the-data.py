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
INPUT_DIR = "../input/"

df_pokemon = pd.read_csv(INPUT_DIR + 'pokemon.csv')

df_battles = pd.read_csv(INPUT_DIR + 'battles.csv')

df_test = pd.read_csv(INPUT_DIR + 'test.csv')
df_pokemon.describe(include='all')
df_battles.describe(include='all')
df_test.describe(include='all')
df_pokemon.describe(include='all').plot.bar(figsize=(21,11))
df_battles.describe(include='all').plot.bar(figsize=(21,11))