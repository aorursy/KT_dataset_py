project_name = 'movielens-eda'
!pip install jovian --upgrade -q

!pip install bar-chart-race -q
import jovian

import numpy as np; import pandas as pd; import matplotlib.pyplot as plt

import seaborn as sns

import bar_chart_race as bcr

import pycountry             

import operator

from collections import Counter

import warnings     



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
warnings.filterwarnings('ignore')

movies = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv')

pd.options.display.max_columns = 30

pd.set_option('display.float_format', '{:,}'.format) #to display float with commas
movies.set_index('id', inplace=True)

movies.head(3)
print(f'Number of rows: {movies.shape[0]}')

print(f'Number of columns: {movies.shape[1]}')
movies.drop(movies[movies['adult'] == 'True'].index, axis='rows', inplace=True)

movies.drop(labels = ['adult', 'belongs_to_collection', 'homepage', 'poster_path', 'video'], axis='columns', inplace=True)
movies.columns

movies = movies.reindex(columns = ['imdb_id','title','original_title','release_date','overview','tagline','genres', 'runtime',

                                   'original_language','spoken_languages','production_companies','production_countries', 

                                   'budget', 'revenue','status',  'vote_average','vote_count','popularity',])

movies.head(3)
movies[['title', 'genres', 'production_companies', 'production_countries', 'spoken_languages']].head(3)
import re



regex = re.compile(r": '(.*?)'")

movies['genres'] = movies['genres'].apply(lambda x: ', '.join(regex.findall(x)))
print('Number of missing values in production_companies column: {}'.format(movies['production_companies'].isna().sum()))

print('Number of missing values in spoken_languages column: {}'.format(movies['spoken_languages'].isna().sum()))
movies.dropna(subset=['production_companies'], axis='rows', inplace=True)

movies.dropna(subset=['spoken_languages'], axis='rows', inplace=True)



print('Number of missing values in production_companies column: {}'.format(movies['production_companies'].isna().sum()))

print('Number of missing values in spoken_languages column: {}'.format(movies['spoken_languages'].isna().sum()))
movies['production_companies'] = movies['production_companies'].apply(lambda x: ', '.join(regex.findall(x)))

movies['production_countries'] = movies['production_countries'].apply(lambda x: ', '.join(regex.findall(x)))

movies['spoken_languages'] = movies['spoken_languages'].apply(lambda x: ', '.join(regex.findall(x)))
movies[['title', 'genres', 'production_companies', 'production_countries', 'spoken_languages']].head(3)
movies.info()
movies['budget'] = movies['budget'].astype(float)

movies['popularity'] = movies['popularity'].astype(float)

movies['release_date'] = pd.to_datetime(movies['release_date'], format='%Y/%m/%d', errors='coerce')

movies['runtime'] = pd.to_timedelta(movies['runtime'], unit='m')
movies.info()
print('Number of missing values in imdb_id: {}'.format(movies['imdb_id'].isna().sum()))

movies.dropna(subset=['imdb_id'], inplace=True)

print('Number of missing values in imdb_id after drop: {}'.format(movies['imdb_id'].isna().sum()))
cond = movies['imdb_id'].duplicated(keep=False)

movies.loc[cond, ['imdb_id','title','release_date', 'overview']].sort_values('imdb_id').head(10)
print('Number of duplicate imdb_ids before drop: {}'.format(movies['imdb_id'].duplicated().sum()))

movies.drop_duplicates('imdb_id', inplace=True)

print('Number of duplicate imdb_ids remaining: {}'.format(movies['imdb_id'].duplicated().sum()))
cond = movies['title'].duplicated(keep=False)

movies.loc[cond, ['imdb_id','title','release_date', 'overview']].sort_values('title').head(10)
movies.describe().T
runtime_int = movies['runtime']/np.timedelta64(1, 'm')



with plt.style.context('seaborn'):

    fig = plt.figure(figsize=(20,20));

    grid = plt.GridSpec(3, 2, wspace=0.2, hspace=0.2);

    plt.rc(('xtick', 'ytick'), labelsize=15); plt.rc('axes', labelsize=15); plt.rcParams["patch.force_edgecolor"] = True;

    _ = plt.subplot(grid[0, 0:]); _ = sns.distplot(runtime_int, kde=False, axlabel='Runtime in Minutes');

    _ = plt.subplot(grid[1,0]); _ = sns.distplot(movies['budget'], kde=False, axlabel='Budget in USD Millions');

    _ = plt.subplot(grid[1,1]); _ = sns.distplot(movies['revenue'], kde=False, axlabel='Revenue in USD Billions');

    _ = plt.subplot(grid[2,0]); _ = sns.distplot(movies['vote_average'], kde=False, axlabel='Vote Average');

    _ = plt.subplot(grid[2,1]); _ = sns.distplot(movies['vote_count'], kde=False, axlabel='Vote Count');

cols = ['title', 'budget']

budget_df = movies.sort_values('budget', ascending=False)[cols].set_index('title')

top_10_budget = budget_df.head(10)



fig, ax = plt.subplots(figsize=(15,5))

sns.set_style('dark')

sns.barplot(data=top_10_budget, x=top_10_budget.index, y='budget');

plt.xticks(ha='left', rotation=-20, fontsize=15); plt.yticks(fontsize=15)

plt.xlabel(''); plt.ylabel('USD 100 Million', fontsize=15);

plt.title('Top 10 Highest Budget Movies', fontsize=15);
cols = ['title', 'revenue']

revenue_df = movies.sort_values('revenue', ascending=False)[cols].set_index('title')

top_10_revenue = revenue_df.head(10)



fig, ax = plt.subplots(figsize=(15,5))

sns.set_style('dark')

sns.barplot(data=top_10_revenue, x=top_10_revenue.index, y='revenue');

plt.xticks(ha='left', rotation=-20, fontsize=15); plt.yticks(fontsize=15)

plt.xlabel(''); plt.ylabel('USD Billion', fontsize=15);

plt.title('Top 10 Highest Revenue Movies', fontsize=15);
profits_ser = movies['revenue'] - movies['budget']

profits_ser.name = 'profit'

profits_df = movies.join(profits_ser)[['title', 'budget', 'revenue', 'profit']].sort_values('profit', ascending=False)

top_10_profits = profits_df.head(10).set_index('title')



plt.style.use('ggplot')

top_10_profits.plot(kind='bar', figsize=(20,4), fontsize=20)

plt.ylabel('USD Billion', fontsize=20); plt.xlabel('')

plt.xticks(rotation=-20, ha='left')

plt.suptitle('Budget, Revenue and Profit for the Top 10 Profitable Movies', fontsize=20)

plt.axes().legend(fontsize=16);
profits_ser_perc = (top_10_profits['profit'] / top_10_profits['budget'] * 100)

profits_ser_perc = profits_ser_perc.sort_values(ascending=False).to_frame().rename(columns={0:'Profit Percentage'})



fig, ax = plt.subplots(figsize=(15,5))

sns.set_style('dark')

sns.barplot(data=profits_ser_perc, x=profits_ser_perc.index, y='Profit Percentage')

plt.xticks(ha='left', rotation=-20, fontsize=15); plt.yticks(fontsize=15)

plt.xlabel(''); plt.ylabel('Profit Percentage', fontsize=15);

plt.title('Profit in Percentage for the Top 10 Profitable Movies', fontsize=15);
top_10_loss = profits_df[profits_df['revenue'] > 0].tail(10).sort_values(['profit', 'revenue']).set_index('title')



plt.style.use('seaborn')

top_10_loss.plot(kind='bar', figsize=(20,6), fontsize=20)

plt.ylabel('USD Billion', fontsize=20); plt.xlabel('')

plt.xticks(rotation=-20, ha='left')

plt.suptitle('Top 10 Losing Movies', fontsize=20)

plt.axes().legend(fontsize=15);
cond = (movies.vote_count)>5000

ratings = movies.loc[cond, ['title', 'vote_average']].sort_values('vote_average', ascending=False)

top_10_ratings = ratings.head(10)



plt.style.use('seaborn')

_ = top_10_ratings.plot(kind='scatter', x='title', y='vote_average', grid=True,

                        figsize=(20,5), fontsize=15, xticks='', ylim=(8,8.6), s=100, 

                        c=['r', 'g', 'y', 'b', 'r', 'b', 'g', 'y', 'r', 'y',]);

for i, (title, vote) in enumerate(zip(top_10_ratings.title, top_10_ratings.vote_average)):

    _ = plt.text(i, vote, title, rotation=-10, va='top', ha='left', 

                 fontsize=15, fontfamily='fantasy');

plt.ylabel('Vote Average', fontsize=20, fontfamily='fantasy');

plt.title('Top 10 Ratings', fontsize=20, fontfamily='fantasy')

plt.xlabel('');
credits = pd.read_csv('../input/the-movies-dataset/credits.csv')

credits.head()
import re



cast_regex = re.compile(r"'name': '(.*?)'")

director_regex = re.compile(r"'Director', 'name': '(.*?)'")

credits['cast'] = credits['cast'].apply(lambda x: ', '.join(cast_regex.findall(x)))

credits['director'] = credits['crew'].apply(lambda x: ', '.join(director_regex.findall(x)))
credits.head()
cond = credits.director == ''

directors = credits.loc[~cond, ['id', 'director']] #List of directors without ''



cond2 = movies.genres == ''

genres = movies.loc[~cond2, 'genres'].reset_index() #Movie genres without ''

genres['id'] = genres['id'].astype(int) #Changing id to from obj to int for merging



director_genre = pd.merge(genres, directors, on='id') #Merging

director_genre.head(5)
genre_dummies = director_genre.genres.str.get_dummies(sep=', ') #Creating dummy variables for genres from director_genre

director_genre_dummies = director_genre.join(genre_dummies) #Joining director_genre with genre_dummies

director_genre_dummies.head(3)
famous_directors = ['Martin Scorsese', 'Quentin Tarantino', 'Steven Spielberg', 'Alfred Hitchcock', 'Christopher Nolan',

                   'Tim Burton', 'James Cameron', 'Ridley Scott', 'George Lucas', 'Woody Allen', 'Clint Eastwood',

                   'Michael Bay', 'Guillermo del Toro', 'John Carpenter', 'Oliver Stone', 'Anurag Kashyap',

                   'Satyajit Ray', 'Mani Ratnam', 'Yash Chopra', 'Rajkumar Hirani', 'Prakash Jha', 'Karan Johar',

                   'S. Shankar', 'Mahesh Bhatt', 'Imtiaz Ali', 'A.R. Murugadoss'] # a small list of famous directors(google)



director_genre_totals = director_genre_dummies.groupby('director').sum() #summing the genres for each director with groupby

cond = director_genre_totals.index.isin(famous_directors) #checking if famous director are in director_genre_totals

famous_director_genres = director_genre_totals[cond] #filter for famous directors

famous_director_genres.drop('id', axis='columns', inplace=True) #dropping id as it is not required

famous_director_genres.head(3)
plt.figure(figsize=(30,10))

sns.set(font_scale=1.7)

sns.heatmap(famous_director_genres.T, cmap='gist_gray', annot=True)



plt.title('Directors and their Genre Counts')

plt.xlabel('Directors')

plt.xticks(rotation=-40, ha='left');
df = credits[['director', 'cast']] #create a dataframe of directors and actors from credits dataframe

df['cast'] = df.groupby('director')['cast'].transform(lambda cast: ' '.join(cast)) #groubpy director and transform cast into one giant string of all cast

df = df.drop_duplicates() #to drop duplicates created by the previous line

df.head()
def cast_count_sort(cast_names):

    d = dict(Counter(cast_names)) #creates a dict with counts of cast names as key, value pairs

    sorted_d = dict(sorted(d.items(), key=operator.itemgetter(1),reverse=True)) #sort the dict in descending

    top_2 = list(sorted_d.items())[:5] #create a list of the sorted dict and slice the first 5

    return top_2



df['cast'] = df['cast'].str.split(', ').apply(cast_count_sort) #aplly cast_count_sort function to cast

df.head()
cond = df.director.isin(famous_directors)

director_actor = df[cond] #filter for famous directors

director_actor = director_actor.set_index(np.arange(len(director_actor))) #set index to len of dataframe, used for merge

actor = pd.DataFrame(director_actor.cast.tolist(), 

                     columns=['actor_1', 'actor_2', 'actor_3', 

                              'actor_4', 'actor_5']) #create a dataframe of actors with individual columns

pd.merge(director_actor, actor, left_index=True, right_index=True).set_index('director').drop('cast', axis='columns')
def get_language(short_code):                                   #a function to extract language from language short-code.

    language = pycountry.languages.get(alpha_2=str(short_code))

    if language:

        return language.name

    else:

        return np.nan

top10_lang_rev = movies.groupby('original_language').sum()['revenue'].sort_values(ascending=False)[:10].to_frame()

top10_lang_rev['language'] = top10_lang_rev.index.map(get_language).fillna('Chinese') 

top10_lang_rev
cond = (movies.original_language.isin(top10_lang_rev.index)) & (movies.vote_count>100)

title_vote_avg = movies.loc[cond,['title', 'vote_average', 'original_language']]



grouped = title_vote_avg.groupby('original_language')

title_vote_avg['vote_rank'] = grouped['vote_average'].rank(method='first').astype(int)

title_vote_avg['language'] = title_vote_avg['original_language'].apply(get_language).fillna('Chinese')



lang_top10 = title_vote_avg[title_vote_avg['vote_rank']<11]

lang_top10.pivot(['language', 'original_language'], 'vote_rank', 'title')
cond = (movies.status == 'Released') & (movies.genres != '') & (movies.release_date.notna()) #filter for releaesed, non-empty genres missing release dates

released_movies = movies[cond] 

release_year = pd.to_datetime(released_movies.release_date) #retrieving year from released_movies
genre_dummies = released_movies.genres.str.get_dummies(sep=', ').set_index(release_year).sort_index() #creating genre dummy variables

genre_yearly = genre_dummies.resample('Y').sum() #create yearly totals

genre_yearly = genre_yearly.loc[(genre_yearly != 0).any(axis=1)] #remove rows which are all zeros, cause by yearly resampling.
bcr.bar_chart_race(genre_yearly, period_fmt='%Y', fixed_order=False, n_bars=5,

                   interpolate_period=False, period_length=2000,

                  steps_per_period=20) 
jovian.commit(project=project_name, environment=None)