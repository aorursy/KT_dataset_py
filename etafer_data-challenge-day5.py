# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

pokemons=pd.read_csv("../input/Pokemon.csv")

# Any results you write to the current directory are saved as output.
pokemons.describe()

print(pokemons)
waterPoke=pokemons["HP"][pokemons["Type 1"]=="Water" ]

waterPoke.append(pokemons["HP"][pokemons["Type 2"]=="Water" ])

print(waterPoke)
scipy.stats.chisquare(pokemons["Type 1"].value_counts())
scipy.stats.chisquare(pokemons["Generation"].value_counts())
contingencyTable = pd.crosstab(pokemons["Type 1"],

                              pokemons["Generation"])

scipy.stats.chi2_contingency(contingencyTable)
pokemons.hist()

plt.show()
import numpy

dat1=pd.DataFrame(pokemons, columns=['Total','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed'])

correlations=dat1.corr()

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(correlations, vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = numpy.arange(0,9,1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(dat1.columns)

ax.set_yticklabels(dat1.columns)

plt.show()
dat2=pd.DataFrame(pokemons, columns=['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed'])

dat2.plot.box()
dat2.plot.barh(stacked=True);
dat3=pd.DataFrame(pokemons, columns=['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed'])

dat3.plot.area(stacked=False)