import pandas as pd

import numpy as np
!ls -l ../input/
df = pd.read_csv("../input/dsm-beuth-edl-demodata-dirty.csv")

df
df.dropna()
df = df.dropna(how='all')

df
df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(0)

df

df['age'] = df['age'].map(lambda age: int(age) if age > 0 else -1*int(age))

df
df.drop(['id'], inplace=True, axis=1)

df
df = df.drop_duplicates(subset=['full_name', 'email'])

df
df['email'] = df['email'].fillna('')

df