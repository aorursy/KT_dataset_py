import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from collections import Counter
from wordcloud import WordCloud
# set plotly credentials to run plotly graphs
movie_init_df = pd.read_csv("../input/movie_metadata.csv")
movie_init_df.shape
movie_init_df.head(5)
top_20_movies = movie_init_df.sort_values('imdb_score', ascending = False)[['movie_title','imdb_score']][:20]
movie_rating_count = []
movie_rating = []
for i in pl.frange(1,9.5,0.5):
    movie_rating_count.append(len(movie_init_df.imdb_score[(movie_init_df['imdb_score'] >= i) & (movie_init_df['imdb_score'] < i+0.5)]))
    movie_rating.append(i)
plt.figure(figsize = (10,10))
plt.title("IMDB score distribution ")
plt.ylabel("IMDB rating ")
plt.xlabel('Frequency')
plt.barh(movie_rating,movie_rating_count, height = 0.4, tick_label = movie_rating)
plt.show()
# The ratings of the movies are normally distributed with description as follows:
movie_init_df['imdb_score'].describe()
# Top 20 movies with respect to ratings
plt.figure(figsize=(10,10))
plt.title("Top 20 movies ")
plt.ylabel("IMDB rating ")
plt.xlabel('Movie Title')
plt.bar(np.arange(len(top_20_movies.movie_title)), list(top_20_movies.imdb_score), alpha = 0.5)
plt.xticks(range(len(top_20_movies.movie_title)),list(top_20_movies.movie_title),rotation=90,fontsize=10)
plt.show()
# Correlation among various varibales
correlation=movie_init_df.corr()
correlation
plt.figure(figsize=(10,10))
tmp=plt.matshow(correlation,fignum=1)
plt.xticks(range(len(correlation.columns)),correlation.columns,rotation=90,fontsize=10)
plt.yticks(range(len(correlation.columns)),correlation.columns,fontsize=10)
plt.colorbar(tmp,fraction=0.05)
plt.show()
# Cast total facebook likes majority comprises of Actor 1 likes. After Actor 1 facebook likes total facebook likes is dominated by Actor 2 likes and then Actot 3 likes
# Number of critic for review is majorily dominated by movie facebook likes
# IMDB score is affected majoorily by num of users for reviews and number of voted users.
# IMDB score is also affected by duration of movie and is negatively correlated with year. It can be said that in past there has been goofd movies.
def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
# Genre count analysis
genre_list = movie_init_df.genres
genre=[w
       for t in genre_list
           for w in t.split('|')]
genre_count = Counter(genre)
genre_df = pd.DataFrame({'Genre':list(genre_count.keys()), 'Count':list(genre_count.values())})
top_genre_df = genre_df.sort_values('Count',ascending=False)[:20]
#WordCloud
show_wordcloud(str(genre))
# wordcloud showcases the various genres. Comedy, Crime, Drama and Romance are the most common genres.
top_genres = np.array(top_genre_df.Genre)
avg_genre_rating = []
for g in top_genre_df.Genre:
    avg_genre_rating.append(movie_init_df.imdb_score[movie_init_df.genres.str.contains(g)])



top_genres=np.insert(top_genres,0,'')
plt.figure(figsize = (12,8))
plt.title("IMDB Score Vs Genre")
plt.ylabel("IMDB Score")
plt.xlabel('Genre')
plt.boxplot(avg_genre_rating, widths=0.5)
plt.xticks(range(len(top_genres)),top_genres, rotation=90)
plt.show()
# Average rating of all genres varies from 6.5 to 7.5.
# Movie related to Drama, Comedy and Action has highly varied ratings.
# analysis with country
country_df = Counter(movie_init_df.country)
country_df = pd.DataFrame({'Country':list(country_df.keys()), 'Count':list(country_df.values())})
top_country_df = country_df.sort_values('Count',ascending=False)[:10]
country_list = np.array(top_country_df.Country)
country_score=[]

country_list = np.array(top_country_df.Country)
country_score=[]
for i in country_list:
    country_score.append(movie_init_df.imdb_score[movie_init_df.country==i])
country_list=np.insert(country_list,0,'')
plt.figure(figsize=(12,8))
plt.title("IMDB Score Vs Country")
plt.ylabel("IMDB Score")
plt.xlabel('Country')
plt.boxplot(country_score,widths = 0.5)
#plt.bar(np.arange(len(country_list)), country_score, alpha = 0.5)
plt.xticks(range(len(country_list)),country_list,rotation=90,fontsize=8)
plt.show()
# India, Italy and USA are relatively high on movies number and average imdb score.
# USA and Canada produces movies of top ratings. Wesern countries are producing good quality movies.
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#Heatmap analysis on average imdb rating
def world_map(score,country_code):
    
#==============================================================================
#     scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
#             [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]
#==============================================================================
    data =[ dict(
                type = 'choropleth',
                locations = country_code,
                z = score,
                text = country_code,
                colorscale = [[-1,"rgb(5, 10, 172)"],[-0.5,"rgb(40, 60, 190)"],[0.0,"rgb(70, 100, 245)"],\
                    [0.3,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
                autocolorscale = False,
                reversescale = True,
                marker = dict(
                    line = dict (
                        color = 'rgb(180,180,180)',
                        width = 0.5
                    ) ),
                colorbar = dict(
                    autotick = False,
                    title = 'Polarity')
                      )]
    layout = dict(
            title = 'World Map Plot',
            geo = dict(
                showframe = False,
                showcoastlines = True,
                projection = dict(
                    type = 'Mercator'
                )
                )   
            )
    
    fig = dict(data = data, layout = layout)
    return fig
country_df = movie_init_df[['country','imdb_score']]
country_df = country_df.groupby('country').mean()

import pycountry as pyc
world_dict = dict()
world_dict['']=''
world_dict['USA'] = 'USA'
for country in pyc.countries:
    country_code = country.alpha_3
    country_name = country.name
    world_dict[country_name] = country_code

country_list = []
for i in country_df.index:
    try:
        country_list.append(world_dict[i])
    except KeyError:
        country_list.append('')
fig = world_map(country_df.imdb_score,country_list)
py.iplot(fig, validate =False)
# Heat map shows the average ratings across the various countries of the world.
# analysis with director and facebook likes
# Top 20 directors based on facebook popularity and their imdb score distribution
director_list = movie_init_df.director_name.unique()
facebook_likes = []
for d in director_list:
    facebook_likes.append(movie_init_df.director_facebook_likes[movie_init_df.director_name == d].mean())

popular_director_df = pd.DataFrame({'Director':director_list,'Facebook_likes':facebook_likes})
popular_director_list = np.array(popular_director_df.sort_values('Facebook_likes',ascending = False)['Director'].head(20))
director_score = []
for d in popular_director_list:
    director_score.append(movie_init_df.imdb_score[movie_init_df.director_name==d])
popular_director_list=np.insert(popular_director_list,0,'')
plt.figure(figsize=(15,10))
plt.title("IMDB Score Vs Popular Director (based on FB likes)")
plt.ylabel("IMDB Score")
plt.xlabel('Director')
plt.boxplot(director_score,widths = 0.5)
plt.xticks(range(len(popular_director_list)),popular_director_list,rotation=90,fontsize=8)
plt.show()
# Based on FB likes, top directors produces few movies with considerable good average rating.
# As a director produces more movies its popularity goes down and but average rating is good.
# analysis with critic reviews
critic_reviews = movie_init_df[['num_critic_for_reviews','imdb_score']]
critic_reviews = critic_reviews.dropna()
plt.figure(figsize=(10,10))
plt.title("IMDB Score Vs Number of Critic Reviews")
plt.ylabel("IMDB Score")
plt.xlabel('Number of Reviews')
plt.scatter(critic_reviews.num_critic_for_reviews,critic_reviews.imdb_score)
plt.show()
# As the number of reviews are increasing ratings are better.
# Plot keywors analysis
plotKeywords_list = movie_init_df.plot_keywords
plotKeywords_list = plotKeywords_list.dropna()
plotKeywords=[w
       for t in plotKeywords_list
           for w in t.split('|')]
plotKeywords_count = Counter(plotKeywords)
plotKeywords_df = pd.DataFrame({'Keywords':list(plotKeywords_count.keys()),'Count':list(plotKeywords_count.values())})
top_plotKeywords_df = plotKeywords_df.sort_values('Count', ascending = False).head(20)
keyword_score = []
tmp_movie_df = movie_init_df[['plot_keywords','imdb_score']]
tmp_movie_df = tmp_movie_df.dropna()
for k in top_plotKeywords_df.Keywords:
    keyword_score.append(tmp_movie_df.imdb_score[tmp_movie_df.plot_keywords.str.contains(k)].mean())
tmp_movie_df = pd.DataFrame({'Keywords':list(top_plotKeywords_df.Keywords), 'Average_score':keyword_score})
tmp_movie_df = tmp_movie_df.sort_values('Average_score' ,ascending = False)
plt.figure(figsize = (10,10))
plt.xticks(range(len(tmp_movie_df.Keywords)),tmp_movie_df.Keywords,rotation = 90)
plt.bar(range(len(tmp_movie_df.Keywords)),tmp_movie_df.Average_score)
plt.show()
# The bar chart shows average imdb rating with the plot keywords.
# People are more interested with the movies related to these topics in genral.
# score comparison with year
date_df = movie_init_df[['title_year','imdb_score']]
date_df = date_df.groupby('title_year').mean()
date_list = date_df.index
d_list = []
for d in date_list:
    d_list.append(str(int(d)))
plt.figure(figsize=(12,12))
plt.title("Average IMDB Score Vs Title Year")
plt.ylabel("Average IMDB Score")
plt.xlabel('Title Year')
#plt.xticks(range(len(d_list)),d_list,rotation = 90)
plt.bar(date_df.index,date_df.imdb_score)
plt.show()
# Average rating is going down with the year. It is due to increase in number of movies.
date_df = movie_init_df[['title_year','imdb_score']]
date_df = date_df.groupby('title_year').max()
date_list = date_df.index
d_list = []
for d in date_list:
    d_list.append(str(int(d)))
plt.figure(figsize=(12,12))
plt.title("Max IMDB Score Vs Title Year")
plt.ylabel("Max IMDB Score")
plt.xlabel('Title Year')
#plt.xticks(range(len(d_list)),d_list,rotation = 90)
plt.bar(date_df.index,date_df.imdb_score)
plt.show()
# Maximum rating is constant with change in year.
# Facenumber vs IMDB score
face_df = movie_init_df[['facenumber_in_poster','imdb_score']]
plt.figure(figsize=(12,12))
plt.title("IMDB Score Vs Face Number in poster")
plt.ylabel("IMDB Score")
plt.xlabel('Face number in poster')
plt.scatter(face_df.facenumber_in_poster, face_df.imdb_score)
plt.show()
# Facenumber increase rating decreases overall
# Overall cast popularity
cast_df = movie_init_df[['cast_total_facebook_likes','imdb_score']]
plt.figure(figsize=(12,12))
plt.title("IMDB Score Vs Cast overall popularity (FB likes)")
plt.ylabel("IMDB Score")
plt.xlabel('Cast Facebook likes')
plt.scatter(cast_df.cast_total_facebook_likes, cast_df.imdb_score)
plt.show()
# Language word cloud
language_list = [str(l) for l in movie_init_df.language.dropna()]
show_wordcloud(language_list)
# Movie with highest profits
budget_df = movie_init_df[['budget','gross','imdb_score','director_name','director_facebook_likes',
                           'movie_facebook_likes','genres','movie_title']]
budget_df = budget_df.dropna()
budget_df['Profit'] = ((budget_df['gross'] - budget_df['budget'])/budget_df['budget'])
budget_df_tmp = budget_df.sort_values('Profit',ascending = False).head(20)
plt.figure(figsize=(10,10))
plt.title("Profit % (Top 20 profitable movies) ")
plt.ylabel("Profit % *100 ")
plt.xlabel('Movie Title')
plt.bar(np.arange(len(budget_df_tmp.movie_title)), list(budget_df_tmp.Profit), alpha = 0.5)
plt.xticks(range(len(budget_df_tmp.movie_title)),list(budget_df_tmp.movie_title),rotation=90,fontsize=10)
plt.show()

# Movie with highest gross income
budget_df_tmp = budget_df.sort_values('gross',ascending = False).head(20)
plt.figure(figsize=(10,10))
plt.title("Gross Income of top 20 movies ")
plt.ylabel("Gross Income ")
plt.xlabel('Movie Title')
plt.bar(np.arange(len(budget_df_tmp.movie_title)), list(budget_df_tmp.gross), alpha = 0.5)
plt.xticks(range(len(budget_df_tmp.movie_title)),list(budget_df_tmp.movie_title),rotation=90,fontsize=10)
plt.show()