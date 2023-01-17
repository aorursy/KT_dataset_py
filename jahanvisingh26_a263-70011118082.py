import pandas as pd
df = pd.read_csv('../input/healthcare/heart.csv', index_col=['age'])
df.head(3)
df.iloc[3]
df.loc[63]
df.loc[[63, 37, 41]]
df.iloc[[2, 1, 0]]
df[:3]
df[3:6]
df['chol'].head(3)
df.chol.head(3)
df.columns = [col.replace(' ', '_').lower() for col in df.columns]

print(df.columns)
df[['cp', 'chol']][:3]
df.cp.iloc[2]
df.cp.iloc[[2]]
(df.cp == 1).head(3)
df[df.cp == 1]
df[(df.chol > 60) | (df.fbs > 10**6)].head(3)
df[df['park_name'].str.split().apply(lambda x: len(x) == 3)].head(3)
df[df.cp.isin([3, 2])].head()
df[df['chol'] < 300]['chol'].plot.hist()
df['cp'].value_counts().sort_index().plot.line()
df['chol'].value_counts().sort_index().plot.area()
df['cp'].value_counts().head(10).plot.bar()