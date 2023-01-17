import pandas as pd
df = pd.read_csv('../input/health-care-data-set-on-heart-attack-possibility/heart.csv')
df.head(3)
df.iloc[2]
df.loc[5]
df.loc[[0, 1, 4]]
df.iloc[[2, 1, 0]]
df[:3]
df[3:6]
df['sex'].head(3)
df.sex.head(3)
df.trestbps
df.columns = [col.replace(' ', '_').lower() for col in df.columns]
print(df.columns)
df[['age', 'chol']][:3]
df.sex.iloc[2]
df.sex.iloc[[2]]
(df.chol == 236).head(3)
df[df.chol == 236]
df[(df.thalach > 60) | (df.trestbps > 10**6)].head()

#df[df['trestbps'].split().apply(lambda x: len(x) == 3)].head(3)

df[df.sex.isin([1])].head()
df['sex'].value_counts().head(10).plot.bar()
(df['sex'].value_counts().head(10) / len(df)).plot.bar()
df['chol'].value_counts().head(10).sort_index().plot.bar()
df['chol'].value_counts().sort_index().plot.line()
df['chol'].value_counts().sort_index().plot.area()
df[df['age'] < 55]['age'].plot.hist()
df['age'].plot.hist()
df[df['chol'] > 234]
df['chol'].plot.hist()