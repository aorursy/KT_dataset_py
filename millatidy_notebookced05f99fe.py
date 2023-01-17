import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import chi2_contingency



dataframe = pd.read_csv('../input/database.csv')
dataframe.head()
dataframe.columns.values
aircraft = dataframe['Aircraft']

aircraft.head()
animal = dataframe['Species Name']

animal.head()
# plt.figure()

# plt.subplot(2,2,1)

# sns.distplot(aircraft, kde=False)



# plt.figure()

# plt.subplot(2,2,2)

# sns.distplot(animal, kde=False)

cross_tab = pd.crosstab(aircraft, animal)

chi2_contingency(cross_tab)