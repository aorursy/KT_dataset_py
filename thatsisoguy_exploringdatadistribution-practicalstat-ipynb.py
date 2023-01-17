# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
state_data = pd.read_csv("../input/murder-rates-by-states/state.csv")
state_data.info()
state_data['Murder.Rate'].quantile([.05, .25, .5, .75, .95])
state_data.Population = state_data.Population/1000000

state_data.boxplot(column = ['Population'], )
import seaborn as sns

ax = sns.boxplot(y = state_data["Population"])
state_data.Population = state_data.Population * 1000000

pd.DataFrame(pd.Series.value_counts(state_data.Population, bins = 11))
from matplotlib import pyplot as plt

plt.hist(state_data['Murder.Rate'], color = 'blue', edgecolor = 'black')
state_data['Murder.Rate'].hist()

state_data.hist()
import seaborn as sns

sns.distplot(state_data['Murder.Rate'], hist=True, kde=True, color = 'blue')