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

fide_historical = pd.read_csv("../input/top-chess-players/fide_historical.csv")
fide_historical.head()
fide_historical[['rating', 'games', 'birth_year']].corr()
from scipy import stats

pearson_coef, p_value = stats.pearsonr(fide_historical["rating"], fide_historical["birth_year"])



# Pearson coefficient / correlation coefficient - how much are the two columns correlated?

print(pearson_coef)



# P-value - how sure are we about this correlation?

print(p_value)
import seaborn as sns

import matplotlib.pyplot as plt

 

# use the function regplot to make a scatterplot

sns.regplot(x=fide_historical["rating"], y=fide_historical["birth_year"],marker="*")

plt.show()

 

# Without regression fit:

# sns.regplot(x=top50["Energy"], y=top50["Loudness..dB.."], fit_reg=False)

# plt.show()