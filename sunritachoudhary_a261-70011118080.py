import pandas as pd
df = pd.read_csv('../input/heartattack/heart.csv', index_col=['age'])
df.head(3)
df.iloc[2]
df.loc[41]
df.loc[[37,59,63]]
df.iloc[[0,2,4]]
df[:3]
df[3:6]
df['cp'].head(3)
df.cp.head(3)
df.cp
df.columns = [col.replace(' ', '_').lower() for col in df.columns]

print(df.columns)
df[['cp', 'fbs']][:3]
df.cp.iloc[2]
df.cp.iloc[[2]]
(df.cp == 1).head(3)
df[df.cp == 1]
df[(df.chol > 60) | (df.fbs > 10**6)].head(3)
df[df['cp'].str.split().apply(lambda x: len(x) == 3)].head(3)
df[df.cp.isin([0,1])].head()
df['cp'].value_counts().head(10).plot.bar()
(df['chol'].value_counts().head(10) / len(df)).plot.bar()
df[df['thalach'] < 200]['thalach'].plot.hist()
