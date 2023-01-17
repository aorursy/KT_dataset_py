import pandas as pd
df = pd.read_csv('../input/health-care-data-set-on-heart-attack-possibility/heart.csv', index_col=['age'])
df.head(4)
df['thalach'].value_counts().head(10).plot.bar()
(df['thalach'].value_counts().head(10) / len(df)).plot.bar()
df['oldpeak'].value_counts().sort_index().plot.bar()
df['oldpeak'].value_counts().sort_index().plot.line()
df['oldpeak'].value_counts().sort_index().plot.area()
df[df['oldpeak'] < 75]['oldpeak'].plot.hist()
df['thalach'].plot.hist()
df[df['thalach'] > 190]
df['oldpeak'].plot.hist()

df.iloc[1]
df.loc[37]
df.loc[[37,63,41]]
df.iloc[[2, 0, 1]]
df[:3]
df[3:5]
df['State'].head(3)
df.chol.head(3)
df.columns = [col.replace(' ', '_').lower() for col in df.columns]
print(df.columns)
df[['fbs', 'thal']][:3]
df.chol.iloc[2]
df.chol.iloc[[2]]
(df.chol == 204).head(3)
df[df.chol == 204]
df[(df.trestbps > 125) | (df.thalach > 160)].head(3)
df[df['chol'].str.split().apply(lambda x: len(x) == 3)].head(3)
df[df.state.isin(['WA', 'OR', 'CA'])].head()
