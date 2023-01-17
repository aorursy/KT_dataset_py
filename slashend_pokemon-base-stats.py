import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pylab

from pandas import *

from pylab import *

from subprocess import check_output



PokemonTable = pd.read_csv('../input/Pokemon.csv')



print("Base stat total average: " + str(PokemonTable['Total'].mean()))

print("Speed average: " + str(PokemonTable['Speed'].mean()))

print("\nPokemon with the highest BST are as follows:")

PokemonTable[PokemonTable.Total == PokemonTable['Total'].max()]
