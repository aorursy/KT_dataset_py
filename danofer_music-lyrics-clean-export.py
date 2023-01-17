import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv('../input/lyrics.csv',usecols=['song', 'year', 'artist', 'genre', 'lyrics'])



df.head()
#replace carriage returns

df = df.replace({'\n': ' '}, regex=True)

df = df.loc[df.year>1800] # drop 5 songs with bad year

df.year = pd.to_datetime(df.year,format="%Y")
print(df.shape[0])

df.drop_duplicates(subset=["lyrics"],inplace=True)

print(df.shape[0])
df['word_count'] = df['lyrics'].str.split().str.len()

print(df['word_count'].describe())

df.head()
df['word_count'].quantile(0.02)
df['word_count'].quantile(0.98)
df.genre.value_counts()
df.artist.nunique()
df = df[df['word_count'] >= 20]

df = df[df['word_count'] <= 650]



print(df.shape[0])

df.drop(["word_count"],axis=1,inplace=True)
df.to_csv("songLyrics.csv.gz",index=False,compression="gzip",encoding="utf-8")