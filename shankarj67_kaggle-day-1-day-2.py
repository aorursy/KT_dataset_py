#Reading library

import pandas as pd

import numpy as np
#Reading datasets

df = pd.read_csv('../input/database.csv')
# Getting the summary of data

df.describe()
# importing library

import seaborn as sns
# Picking one column with numeric variables

numerical_col = df['Engines']
# filling nan values with 0

numerical_col_fill = numerical_col.fillna(0)
#plotting a histogram

sns.distplot(numerical_col_fill, kde = False).set_title("Engine_plot")