import pandas as pd
df = pd.read_csv('../input/videogamesales/vgsales.csv', index_col=['Name'])
df.head(3)
df.iloc[2]
df.loc['Grand Theft Auto V']
df.loc[['Hitman (2016)', 'Call of Duty: Modern Warfare 3', 'Grand Theft Auto V']]
df.iloc[[2, 1, 0]]
df[:3]
df[3:6]
df['Genre'].head(3)
df.Genre.head(3)
df.EU_Sales
df.columns = [col.replace(' ', '_').lower() for col in df.columns]

print(df.columns)
df[['genre', 'global_sales']][:3]
df.genre.iloc[2]
df.genre.iloc[[2]]
(df.genre == 'Racing').head(3)
df[df.genre == 'Action']
df[(df.year > 2005) & (df.platform == 'PC')].head(3)
df[df.publisher.isin(['Activision', 'Ubisoft', 'Electronic Arts'])].head(3)
df['publisher'].value_counts().head(10).plot.bar()
(df['publisher'].value_counts().head(10) / len(df)).plot.bar()
df['year'].value_counts().sort_index().plot.bar()
df['platform'].value_counts().sort_index().plot.bar()