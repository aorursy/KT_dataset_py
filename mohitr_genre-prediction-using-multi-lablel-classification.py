import pandas as pd 
movie_plots = pd.read_csv('../input/wiki_movie_plots_deduped.csv')
movie_plots.head()
movie_plots.shape
movie_plots.isnull().sum()
#Total unique combinations of genres



len(df['Genre'].unique())
#Checking most popular genres



df = movie_plots.groupby(["Genre"]).size().reset_index(name='count')

df = df.sort_values(by=['count'], ascending=False)

df.head(n=50)
#Keeping only generes with 50+ instances



df = df[df['count']>=50]
df.shape
genre_list = df['Genre'].tolist()

genre_list
#Removing genre 'unknown'



genre_list = genre_list[1:]
movie_plots_ = movie_plots[movie_plots['Genre'].isin(genre_list)]
len(movie_plots_['Origin/Ethnicity'].unique())