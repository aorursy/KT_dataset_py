import pandas as pd
df = pd.read_csv('/Users/Roxanne/Desktop/vle.csv')
df.head(50)
df.iloc[37]
df[:37]
df['activity_type'].head(3)
(df.activity_type == 'resource').head(37)
df['code_module']
import numpy as np
df.shape
df.index
df.columns
df.info
df.count
df.sum
