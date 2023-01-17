import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv('../input/titanic/train.csv')
df.head()
df.isna().sum()
# 'constant_value'

# ffill

# bfill

# df['col'].mean()
df['Embarked'].fillna('S', inplace=True)

df.isna().sum()
# just as same as .fillna(), and is complex
from sklearn.impute import SimpleImputer



imp = SimpleImputer(missing_values=np.nan, strategy='mean')

imp.fit(df[['Age']])

df[['Age']]=imp.transform(df[['Age']])



df.isna().sum()
# axis = 0, drops rows with na 

# axis = 1, drops columns with na
df.dropna(axis=1, inplace=True)

df.isna().sum()