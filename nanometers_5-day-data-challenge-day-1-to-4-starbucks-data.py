# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

sbux = pd.read_csv("../input/starbucks_drinkMenu_expanded.csv")



# Looking at types of data for columns

sbux.info()
# Top 20 rows to see what we're working with data-wise

sbux.head(10)
# Summary statistics for rows, including count / mean / std / min / max and %iles

sbux.describe()
# Day 2 - VISUALS!!!



# Import visualization library

import matplotlib.pyplot as plt

import matplotlib.mlab as mlab

import seaborn as sns



sbux.Beverage_category = sbux.Beverage_category.astype('category')

sbux.Beverage_category.cat.categories



sns.set_style("darkgrid")

m1 = sns.distplot(sbux.Calories, bins = 15)

# Visualizing by beverage category

vis1 = sns.lmplot(data = sbux, x='Calories', y=' Sugars (g)', fit_reg = False, hue = 'Beverage_category', size = 7, aspect = 1)
# ------ Day 3 - t test ----------



from scipy import stats

#sbux.Beverage = sbux.Beverage.astype('category')

#sbux.Beverage.cat.categories



sbux.Beverage_prep = sbux.Beverage_prep.astype('category')

sbux.Beverage_prep.cat.categories

lite = sbux[sbux.Beverage_category == 'Frappuccino® Light Blended Coffee'].Calories

reg = sbux[sbux.Beverage_category == 'Frappuccino® Blended Coffee'].Calories



sns.set_style("darkgrid")

plt.hist(reg, bins = 5)

plt.hist(lite, bins = 5)

plt.title('Light Frapp vs Normal Frapp Calories')

plt.ylabel("Number of Drinks", fontsize = 12, color = "Black")

plt.xlabel("Calories", fontsize = 12, color = "Black")

plt.show()



stats.ttest_ind(lite, reg, equal_var = False)
# ----- Day 4 More Visualizations: Categorical  --------

import seaborn as sns

import pandas as pd

import matplotlib as plt



sns.countplot(y = sbux.Beverage_category)