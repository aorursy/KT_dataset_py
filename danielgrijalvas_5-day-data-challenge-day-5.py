# 5 Day Data Challenge, Day 5

# Using a Chi-Square test



import matplotlib.pyplot as plt

import pandas as pd

from scipy import stats



# read movies.csv 

file = '../input/movies.csv'

data = pd.read_csv(file, encoding='latin1')



data.head()
contingency_table = pd.crosstab(data['year'], data['rating'])
contingency_table
# chi-squared test (two-way)

stats.chi2_contingency(contingency_table)