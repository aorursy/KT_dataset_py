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

economic_freedom = pd.read_csv("../input/the-economic-freedom-index/economic_freedom_index2019_data.csv", encoding = "latin-1")
economic_freedom.head(10)
trade_freedom = []



for x in economic_freedom["Trade Freedom"].iteritems():

    temp = x[1]

    if  np.isnan(temp):

        continue

    else:

        trade_freedom.append(temp)
world_rank = []



for x in economic_freedom["World Rank"].iteritems():

    temp = x[1]

    if  np.isnan(temp):

        continue

    else:

        world_rank.append(temp)
#look at mismatched length

print(len(world_rank))

print(len(trade_freedom))


# Delete excess data

del trade_freedom[len(trade_freedom)-2: len(trade_freedom)]

from scipy import stats

pearson_coef, p_value = stats.pearsonr(world_rank, trade_freedom)



# Pearson coefficient / correlation coefficient - how much are the two columns correlated?

print(pearson_coef)



# P-value - how sure are we about this correlation?

print(p_value)
economic_freedom.corr()
import seaborn as sns

import matplotlib.pyplot as plt

sns.set()





ax = sns.regplot(x=world_rank, y=trade_freedom,)

plt.ylabel("Trade Freedom", fontsize = 10)

plt.xlabel("World Rank", fontsize = 10)

plt.title("Trade Freedom vs. World Rank", fontsize = 10)
economic_freedom.loc[economic_freedom["Trade Freedom"] < 20]
# libraries

import matplotlib.pyplot as plt

import numpy as np

 

# create data

x = world_rank

y = trade_freedom

z = economic_freedom["GDP Growth Rate (%)"]

 

# use the scatter function

plt.scatter(x, y, s=z*1000, alpha=0.5)

# plt.show()

plt.ylabel("Trade Freedom", fontsize = 10)

plt.xlabel("World Rank", fontsize = 10)

plt.title("Trade Freedom vs. World Rank (with GDP Growth)", fontsize = 10)
economic_freedom.loc[economic_freedom["GDP Growth Rate (%)"] > 70]
import matplotlib.pyplot as plt

# library & dataset

import seaborn as sns

 

# Basic 2D density plot

sns.set_style("white")

sns.kdeplot(world_rank, trade_freedom)

 

plt.ylabel("Trade Freedom", fontsize = 10)

plt.xlabel("World Rank", fontsize = 10)

plt.title("Trade Freedom vs. World Rank", fontsize = 10)

# # Custom it with the same argument as 1D density plot

# sns.kdeplot(df.sepal_width, df.sepal_length, cmap="Reds", shade=True, bw=.15)

 

# # Some features are characteristic of 2D: color palette and wether or not color the lowest range

# sns.kdeplot(df.sepal_width, df.sepal_length, cmap="Blues", shade=True, shade_lowest=True, )

# sns.plt.show(
