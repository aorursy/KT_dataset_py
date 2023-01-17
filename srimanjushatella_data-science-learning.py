# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/pokemon.csv')
data.info()
import matplotlib.pyplot as mpt

import seaborn as sb


f,ax = mpt.subplots(figsize=(18, 18))

sb.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f', ax=ax)
data.head(20)
data.columns
data.Speed.plot(kind = 'line', color = 'b',label = 'Defense', linewidth=1, alpha=1, grid = True, linestyle = '--')

data.Attack.plot(kind = 'line', color = 'g',label = 'Attack', linewidth=1, alpha=0.5, grid = True, linestyle = ':')

mpt.legend(loc = 'upper left')

mpt.xlabel('X axis')

mpt.ylabel('Y axis')

mpt.title('Line plot between attack and defense of pokemons')
data.plot(kind='scatter', x='Defense', y='Attack',alpha = 0.5,color = 'yellow')

mpt.xlabel('Defense')

mpt.ylabel('Attack')

mpt.title('Attack and Defense scatter graph')
data.Speed.plot (kind = 'hist', bins= 100, figsize= (18,18))