# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import datasets

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
iris = pd.read_csv('../input/Iris.csv')
iris.describe
iris.head(5)
# The column ‘Species’ is categorical,  the frequency distribution for each category.

iris['Species'].value_counts()
import statsmodels.api as sm

# Correlation Matrix

# The correlation function uses Pearson correlation coefficient, which results in a number between -1 to 1

# A strong negative relationship is indicated by a coefficient closer to -1 and a strong positive correlation is indicated by a coefficient toward 1

corr = iris.corr()

print(corr)
sm.graphics.plot_corr(corr, xnames=list(corr.columns))

plt.show()
from pandas.tools.plotting import scatter_matrix

# the distribution of the interactions of each pair of attributes

# matrix of scatter plots of all attributes against all attributes

scatter_matrix(iris, figsize=(10, 10))

plt.suptitle("Pair Plot", fontsize=20)