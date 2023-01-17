import pandas as pd
df = pd.read_csv('../input/chai-time-data-science/Episodes.csv',index_col=['episode_id'])
df.head(5)
df.iloc[5]
df.loc['E2']
df.loc[['E1', 'E3', 'E2']]
df.iloc[[2, 1, 0]]
df[:5]
df[5:10]
df['heroes_location'].head(5)
df.heroes_location.head(5)

df.columns = [col.replace(' ', '_').lower() for col in df.columns]
print(df.columns)
df[['heroes_location', 'heroes_nationality']][:5]
df.heroes_location.iloc[5]
df.heroes_location.iloc[[2]]
(df.heroes_nationality == 'india').head(5)
df[df.heroes_nationality == 'USA']
df[(df.spotify_listeners > 100) | (df.apple_listeners > 15)].head(5)

df[df.heroes_nationality.isin(['USA', 'india', 'canada'])].head()
df['heroes_nationality'].value_counts().head(10).plot.bar()
(df['heroes_nationality'].value_counts().head(10) / len(df)).plot.bar()
df['apple_listeners'].value_counts().head(10).sort_index().plot.bar()
df['apple_listeners'].value_counts().sort_index().plot.line()
df['apple_listeners'].value_counts().sort_index().plot.area()
df[df['apple_listeners'] > 10]['apple_listeners'].plot.hist()
df['apple_listeners'].plot.hist()
df[df['apple_listeners'] > 15]
df['apple_listeners'].plot.hist()