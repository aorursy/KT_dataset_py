import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from IPython.display import display
warnings.filterwarnings('ignore') # ignore warnings.

%config IPCompleter.greedy = True # autocomplete feature.

pd.options.display.max_rows = None # set maximum rows that can be displayed in notebook.

pd.options.display.max_columns = None # set maximum columns that can be displayed in notebook.

pd.options.display.precision = 2 # set the precision of floating point numbers.
# # Check the encoding of data. Use ctrl+/ to comment/un-comment.



# import chardet



# rawdata = open('../input/database.csv', 'rb').read()

# result = chardet.detect(rawdata)

# charenc = result['encoding']

# print(charenc)

# print(result) # It's ascii with 100% confidence.
df = pd.read_csv('../input/database.csv', encoding='utf-8')

df.drop_duplicates(inplace=True) # drop duplicates if any.

df.shape # num rows x num columns.
miss_val = (df.isnull().sum()/len(df)*100).sort_values(ascending=False) # columns and their missing values in percentage.

miss_val[miss_val>0]
df.head()
df.groupby('Incident Year').size().plot()
df.groupby('Incident Month').size().plot()
strike = ['Radome Strike', 'Windshield Strike', 'Nose Strike', 'Engine1 Strike', 'Engine2 Strike', 'Engine3 Strike',

          'Engine4 Strike', 'Propeller Strike', 'Wing or Rotor Strike', 'Fuselage Strike', 'Landing Gear Strike',

          'Tail Strike', 'Lights Strike', 'Other Strike']
table = df.groupby('Incident Month')[strike].sum()

table # Incident Month vs Strike.
(table.sum(axis=0) # column sum.

,table.sum(axis=1)) # row sum.
table.sum(axis=0).sum(), table.sum(axis=1).sum() # sum of all rows and all columns.
df.groupby('Incident Month')[strike].sum().min()
from scipy import stats



chi2_stat, p_val, dof, ex = stats.chi2_contingency(table)

print("===Chi2 Stat===")

print(chi2_stat)

print("\n")

print("===Degrees of Freedom===")

print(dof)

print("\n")

print("===P-Value===")

print(p_val)

print("\n")

print("===Contingency Table===")

pd.DataFrame(ex)