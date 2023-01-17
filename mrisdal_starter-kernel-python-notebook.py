# Load modules

import pandas as pd # Data manipulation

import matplotlib.pyplot as plt # Data visualization

import seaborn as sns

%matplotlib inline



# Read in the data

recipes = pd.read_csv('../input/epi_r.csv')
sns.distplot(recipes["rating"])

plt.savefig('rating_distribution.png')