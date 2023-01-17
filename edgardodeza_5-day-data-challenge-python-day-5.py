import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy.stats import chisquare
import scipy

plt.style.use('fivethirtyeight')
dog_owners = pd.read_csv('../input/20151001hundehalter.csv')
dog_owners.head()
contingency_table = pd.crosstab(dog_owners["ALTER"], dog_owners["GESCHLECHT"])
contingency_table
print(scipy.stats.chi2_contingency(contingency_table))
result = scipy.stats.chi2_contingency(contingency_table)

print('\nchi_square =', result[0])
print('p =', result[1])
print('dof =', result[2])
# The following code is copied from
# https://www.kaggle.com/ekeneo/my-5-day-data-challenge-day-5

import seaborn as sns

# make a barplot from two categorical columns in our dataframe
sns.countplot(dog_owners["ALTER"],
              hue=dog_owners["GESCHLECHT"], palette="Set2")

plt.show()