import pandas as pd

import seaborn as sns

from scipy.stats import chisquare
# read the dataset

data = pd.read_csv('../input/sets.csv')



data.head()
# perform chi-square on names

names = data['name'].value_counts()

print(names)

chisquare(names)
# visualize names

sns.distplot(names)