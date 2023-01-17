import numpy as np
import pandas as pd 
from pandas import Series, DataFrame 
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
pop = pd.read_csv('../input/state-population.csv')
areas = pd.read_csv('../input/state-areas.csv')
abb = pd.read_csv('../input/state-abbrevs.csv')
print(pop.head()); print(areas.head()); print(abb.head());
merged = pd.merge(pop, abb, how='outer',
                 left_on='state/region', right_on='abbreviation')
merged = merged.drop('abbreviation', 1)
merged.head()
merged.isnull().any()
merged[merged['population'].isnull()].head()
merged.loc[merged['state'].isnull(), 'state/region']
merged.loc[merged['state'].isnull(), 'state/region'].unique()
merged.loc[merged['state/region'] == 'PR', 'state'] = 'Puerto Rico'
merged.loc[merged['state/region'] == 'USA', 'state'] = 'United States'
merged.isnull().any()
final = pd.merge(merged, areas, on='state', how='left')
final.head()
final.isnull().any()
final['state'][final['area (sq. mi)'].isnull()].unique()
final.dropna(inplace=True) #affect the underlined data
final.head()
data2010 = final.query("year == 2010 & ages == 'total'")
data2010.head()