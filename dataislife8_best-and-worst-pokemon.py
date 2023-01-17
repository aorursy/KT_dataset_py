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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pokedata = pd.read_csv('../input/Pokemon.csv')
pokedata
pokedata.keys()
pokedata = pokedata.drop(['Type 1', 'Type 2', 'Generation', 'Legendary'],1)
pokedata.head()
pokedata = pokedata.drop(['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def'],1)
pokedata = pokedata.drop(['Speed'],1)
pokedata.head()
pokedata_sort = pokedata.sort_values('Total')
pokedata_sort.head()
pokedata_sort.tail()
worst_pokemon = pokedata_sort.head(1)
worst_pokemon
best_pokemon = pokedata_sort.tail(1)
best_pokemon
