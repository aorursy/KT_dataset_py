import numpy as np 

import pandas as pd

import re
df=pd.read_csv('../input/oni-apur-dataset/Manitou10_microbe_taxonomy.csv', sep='|', names=['col1'])

df
df2 = df.col1.str.extractall(r'(?P<name>[A-Z ]+[A-Z]):(?P<value>[A-Z /]+[A-Z])', flags=re.I).reset_index(level=1, drop=True)

df2

print(df2.to_string())
df3 = df2.set_index('name', append=True).unstack(fill_value='')

df3.columns = df3.columns.droplevel()