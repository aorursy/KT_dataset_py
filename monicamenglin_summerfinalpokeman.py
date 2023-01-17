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
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sn
df = pd.read_csv('../input/Pokemon.csv',header=0)
#head()returns first five rows of data
df.head()
print(list(df.columns.values))
grass = df[(df['Type 1'] == "Grass") | (df['Type 2'] == "Grass")]
fire = df[(df['Type 1'] == "Fire") | (df['Type 2'] == "Fire")]

#both = pd.concat((grass['Total'], fire['Total']))
#sn.boxplot(data=df)
#sn.boxplot(data=grass)
sn.boxplot(data=fire)
#print(df.query('Type 1==Grass'))
#print(df.mask('Type 1', 'Grass'))
#datafiltered = df[(df.type1 == 'Grass') | (df.type2 == 'Grass')]

#sn.boxplot(data=df)
sn.boxplot(data=grass)
#sn.boxplot(data=fire)

#total for fire is higher than grass type. speed is also a lot less for grass
#Total power of the five pokemon is from 309 to 625.
#
sn.violinplot(x='Type 1', y='Speed', data=df)
#Flying has very high speed. Grass is slower than water, and fire is faster than both so fire is the fastest starting pokemon probably.
sn.violinplot(x='Type 1', y='Total', data=df)
#The weakest pokemon is a flying type. But on average bug type is the weakest.
#the strongest type overall is dragon type which make sense because they are rare
co= df.corr()
sn.heatmap(co,annot=True, linewidths=1.0)

#total for fire is higher than grass type. speed is also a lot less for grass, so I like fire type more.
#special attack tends to correlate with special defense, which means if your pokemon has high special attack it probably has high special defense. This makes sense because originally pokemon only had a single "special" stat then later they split it into two.
#also legendary pokemon tend to have high special attack and defense comared to other stats

