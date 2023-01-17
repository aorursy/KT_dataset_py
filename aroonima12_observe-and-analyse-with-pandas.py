import pandas as pd

import numpy as np
df=pd.read_csv('../input/chess/games.csv')

df.head(5)
df.info()
df.describe()
df['victory_status'].unique()
df['victory_status'].value_counts()
df['winner'].unique()
df['winner'].value_counts()
group=df.groupby('victory_status')

group.mean()
df['opening_name'].nunique()
df['opening_ply'].nunique()
df['opening_name'].value_counts().head(50)