import seaborn as sns

import numpy as np

import pandas as pd

from scipy import stats
# Load data

dataframe = pd.read_csv('../input/DigiDB_digimonlist.csv')
# Show part of data 

dataframe.head()
# Barplot 

sns.barplot(x = 'Type', y = 'Memory', data = dataframe).set_title('Memory vs. Digimon Type')
# Scatterplot

sns.scatterplot(x = 'Memory', y = 'Lv 50 HP', data = dataframe).set_title('Memory vs. Lv 50 HP')
# Pearson Correlation and P-value of Scatterplot

stats.pearsonr(x = dataframe['Memory'], y = dataframe['Lv 50 HP'])
# Low P-value indications that low probability of null hypothesis being correct, meaning high chance that the correlation is significant. 