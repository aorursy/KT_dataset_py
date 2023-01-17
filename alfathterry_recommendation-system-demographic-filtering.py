import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/demographic.csv')
pd.set_option('display.max_columns', None)
df.head()
genre = ["Animation", "Comedy"] 
duration = (60, 150)
year = (2000, 2019)
topk = 10
def demographic_filter(df, genre=None, duration=None, year=None):
        df = df.copy()
        
        if genre is not None:
            df = df[df[genre].all(axis=1)]
        if duration is not None:
            df = df[df.runtime.between(duration[0], duration[1])]
        if year is not None:
            df = df[df.release_year.between(year[0], year[1])]
        return df
df_filtered = demographic_filter(df, genre = ["Animation", "Comedy"],
                                     duration = (60, 150), 
                                     year = (2000, 2019) )
df_filtered.head()
recommendation = df_filtered.loc[:, :"release_year"] # buang kolom yang tidak perlu, agar lebih rapi
recommendation = recommendation.sort_values('vote_average', ascending=False).head(topk) # Sorting berdasrakan vote_average
recommendation
df['vote_count'].describe()
df['vote_count'].hist(bins=15)
df['vote_count'].quantile(0.925)
C = (df['vote_average'] * df['vote_count']).sum() / df['vote_count'].sum()
C
def imdb_score(df, q=0.925):
    df = df.copy()
    
    m = df['vote_count'].quantile(q)
    C = (df['vote_average'] * df['vote_count']).sum() / df['vote_count'].sum()
    
    df = df[df['vote_count'] >= m]
    df["score"] = df.apply(lambda x: (x['vote_average'] * x['vote_count'] + C*m) / (x['vote_count'] + m), axis=1)
    return df 
df_imdb = imdb_score(df_filtered)
df_imdb.head()
recommendation = df_imdb.loc[:, "title": "release_year"]
recommendation = recommendation.sort_values("vote_average", ascending=False).head(topk)
recommendation