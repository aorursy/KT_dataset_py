# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd
population = pd.read_csv("/kaggle/input/turkey-population-2018/population.csv", sep = ",")
new_index = (population["Population"].sort_values(ascending = False)).index.values

new_index
sorted_population = population.reindex(new_index)

sorted_population
population2 = pd.read_csv("/kaggle/input/turkey-population-2018/AgePopulation.csv", sep = ",")
population2
new_index2 = (population2["Total"].sort_values(ascending = False)).index.values

sorted_population2 = population2.reindex(new_index2)

sorted_population2