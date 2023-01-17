import numpy as np

import pandas as pd

import seaborn as sns

from scipy import stats
# Read data

dataframe = pd.read_csv('../input/DigiDB_movelist.csv')
# See part of the data

dataframe.head()
# Create array for SP Cost and power data

cost_power_array = pd.crosstab(dataframe['SP Cost'], dataframe['Power'])

cost_power_array.sample(10)
# Chi-Squared Test between Cost and Power 

stats.chi2_contingency(cost_power_array)
# Scatterplot

sns.scatterplot(x = 'SP Cost', y = 'Power', data = dataframe).set_title('Memory vs. Lv 50 HP')