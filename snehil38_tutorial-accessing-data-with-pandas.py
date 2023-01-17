import pandas as pd
df = pd.read_csv('../input/chai-time/Episodes.csv', index_col="heroes")
df.head(5)
df.iloc[6]
df.loc['Ryan Chesler']
df.loc[['Edouard Harris', 'Abhishek Thakur', 'NaN']]
df.iloc[[3, 5, 2]]
df[:5]
df[3:6]
df['flavour_of_tea'].head(3)
df.flavour_of_tea.head(3)

df.columns = [col.replace(' ', '_').lower() for col in df.columns]
print(df.columns)
df[['flavour_of_tea', 'episode_name']][:4]
df.episode_name.iloc[6]
df.episode_name.iloc[[4]]
(df.flavour_of_tea == 'Masala Chai').head(3)
df[df.flavour_of_tea == 'Masala Chai']
df[(df.youtube_subscribers > 3) | (df.anchor_plays > 553.0)].head(3)

df[df.heroes_location.isin(['USA', 'Norway', 'France'] )].head()
df['heroes_nationality'].value_counts().head(5).plot.bar()
(df['heroes_nationality'].value_counts().head(7) / len(df)).plot.bar()
df['anchor_plays'].value_counts().head(10).sort_index().plot.bar()
df['spotify_streams'].value_counts().sort_index().plot.line()
df['apple_listened_hours'].plot.hist()
df[df['apple_listeners'] > 10]
df['anchor_thumbnail_type'].plot.hist()
df[df['apple_listeners'] > 15]['apple_listeners'].plot.hist()
df['spotify_listeners'].value_counts().sort_index().plot.area()