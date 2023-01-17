import numpy as np

import pandas as pd
from plotly.offline import iplot, init_notebook_mode

import cufflinks as cf

import plotly.graph_objs as go

# import chart_studio.plotly as py



init_notebook_mode(connected=True)

cf.go_offline(connected=True)



# Set global theme

cf.set_config_file(world_readable=True, theme='pearl')
import seaborn as sns 

import matplotlib.pyplot as plt
movie_df = pd.read_csv('../input/fmovies-most-watched-contents/Movie.csv')

tv_df = pd.read_csv('../input/fmovies-most-watched-contents/TV.csv')
date_mv = pd.to_datetime(movie_df['Date'],format="%Y-%m-%d")

date_tv= pd.to_datetime(tv_df['Date'],format="%Y-%m-%d")
movie_df['Date'] = date_mv

tv_df['Date'] = date_tv

movie_df.set_index("Date", drop=True,inplace=True)

tv_df.set_index("Date", drop=True,inplace=True)
movie_df.head()
tv_df.info()
site_ratings_mv_median = movie_df['USER_REVIEWS_LOCAL'].median()

imdb_ratings_mv_median = movie_df['IMDB'].median()


fig = go.Figure()

fig.add_trace(go.Histogram(x=movie_df['USER_REVIEWS_LOCAL'], name="Local Ratings|Movie"))

fig.add_trace(go.Histogram(x=movie_df['IMDB'], name="IMDB ratings|Movie"))



fig.add_trace(go.Scatter(

    x=[imdb_ratings_mv_median-1.1, site_ratings_mv_median +1.3],

    y=[300, 299],

    text=["IMDB Median","Site Ratings Median",],

    mode="text",

    showlegend=False

#     name=["Site Ratings Mean","IMDB Mean"]

))

fig.add_shape(

        # Line Horizontal

            type="line",

            x0=imdb_ratings_mv_median,

            y0=0,

            x1=imdb_ratings_mv_median,

            y1=350,

            line=dict(

                width=4,

                dash="dot",

            ),

    )

fig.add_shape(

            type="line",

            x0=site_ratings_mv_median,

            y0=0,

            x1=site_ratings_mv_median,

            y1=350,

            line=dict(

                color="LightSeaGreen",

                width=4,

                dash="dashdot",

            ),

    )



fig.update_layout(title="Distribution Plot Local Ratings Vs IMDB |Movies")

# fig.write_image("../Imgs/Distribution_plt_lcl_ratings_mv.png", width=5)
site_ratings_mv = movie_df.sort_values('USER_REVIEWS_LOCAL', ascending=False,ignore_index=True)

site_ratings_mv.drop_duplicates('MOVIE_NAME',inplace=True )

eval_movie_df = site_ratings_mv[['MOVIE_NAME', 'IMDB','USER_REVIEWS_LOCAL','NUMBER_REVIEWS_LOCAL','SITE_RANK']]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
eval_movie_df[['IMDB','USER_REVIEWS_LOCAL']] = scaler.fit_transform(eval_movie_df[['IMDB','USER_REVIEWS_LOCAL']])
movie_names = eval_movie_df.MOVIE_NAME.values.tolist()

imdb_ratings = eval_movie_df.IMDB.values.tolist()

local_ratings = eval_movie_df.USER_REVIEWS_LOCAL.values.tolist()

fig = go.Figure()

fig.add_trace(go.Bar(x=movie_names[:100], y= imdb_ratings[:100], name="IMDB Ratings Top 100 |Movie"))

fig.add_trace(go.Bar(x=movie_names[:100], y= local_ratings[:100], name="Local Ratings Top 100 |Movie"))

fig.update_xaxes(tickangle=-45)

fig.update_layout(xaxis_title ="Movies",

                  yaxis_title="Ratings",

                  title="IMDB Vs Local Ratings Top 100,|Movie | sorted Local Ratings",height=700)

# for saving image use these dimensions

#width=1000, height=700

# fig.write_image("../Imgs/imdb_vs_local_top_100_mv.png")
fig = go.Figure()

fig.add_trace(go.Bar(x=movie_names[-100:], y= imdb_ratings[-100:], name="IMDB Ratings Bottom 100|Movie"), )

fig.add_trace(go.Bar(x=movie_names[-100:], y= local_ratings[-100:], name="Local Ratings Bottom 100|Movie"))

fig.update_xaxes(tickangle=-45)

fig.update_layout(xaxis_title ="Movies", 

                  yaxis_title="Ratings", 

                  title="IMDB Vs Local Ratings Bottom 100, sorted by local Ratings",

                 height=700)

# for saving image use these dimensions

#width=1000, height=700

# fig.write_image("../Imgs/imdb_vs_local_bottom_100_mv.png")
imdb_ratings_mv = movie_df.sort_values('IMDB', ascending=False, ignore_index=True)

imdb_ratings_mv.drop_duplicates('MOVIE_NAME',inplace=True )
eval_movie_df_imdb = imdb_ratings_mv[['MOVIE_NAME', 'IMDB','USER_REVIEWS_LOCAL','NUMBER_REVIEWS_LOCAL','SITE_RANK']]
eval_movie_df_imdb[['IMDB','USER_REVIEWS_LOCAL']] = scaler.fit_transform(eval_movie_df_imdb[['IMDB','USER_REVIEWS_LOCAL']])
movie_names_imdb = eval_movie_df_imdb.MOVIE_NAME.values.tolist()

imdb_ratings_imdb = eval_movie_df_imdb.IMDB.values.tolist()

local_ratings_imdb = eval_movie_df_imdb.USER_REVIEWS_LOCAL.values.tolist()
fig = go.Figure()

fig.add_trace(go.Bar(x=movie_names_imdb[:100], y= imdb_ratings_imdb[:100], name="IMDB Ratings Top 100|Movie"))

fig.add_trace(go.Bar(x=movie_names_imdb[:100], y= local_ratings_imdb[:100], name="Local Ratings Top 100|Movie"))

fig.update_xaxes(tickangle=-45)

fig.update_layout(xaxis_title ="Movies",

                  yaxis_title="Ratings",

                  title="IMDB Vs Local Ratings Top 100 |Movie |sorted IMDb",

                   height=700)



# for saving image use these dimensions

#width=1000, height=700

# fig.write_image("../Imgs/imdb_vs_local_imdb_sort_top_100_mv.png")

fig = go.Figure()

fig.add_trace(go.Bar(x=movie_names_imdb[-100:], y= imdb_ratings_imdb[-100:], name="IMDB Ratings Bottom 100|Movie"), )

fig.add_trace(go.Bar(x=movie_names_imdb[-100:], y= local_ratings_imdb[-100:], name="Local Ratings Bottom 100|Movie"))

fig.update_xaxes(tickangle=-45)

fig.update_layout(xaxis_title ="Movies",

                  yaxis_title="Ratings",

                  title="IMDB Vs Local Ratings Bottom 100|Movie |soted IMDB", 

                  height=700,)



# for saving image use these dimensions

#width=1000, height=700

# fig.write_image("../Imgs/imdb_vs_local_imdb_sort_bottom_100_mv.png")

imdb_ratings_mv.iloc[-12:,[5,9]]
site_ratings_tv_median = tv_df['USER_REVIEWS_LOCAL'].median()

imdb_ratings_tv_median= tv_df['IMDB'].median()
print(site_ratings_tv_median ,imdb_ratings_tv_median)
fig = go.Figure()

fig.add_trace(go.Histogram(x=tv_df['USER_REVIEWS_LOCAL'], name="Local Ratings|TV"))

fig.add_trace(go.Histogram(x=tv_df['IMDB'], name="IMDB ratings|TV"))



fig.add_trace(go.Scatter(

    x=[imdb_ratings_tv_median-2.3, site_ratings_tv_median + 2.4],

    y=[165, 165],

    text=["IMDB Median TV","Site Ratings Median TV",],

    mode="text",

    showlegend=False

))



fig.add_shape(

            type="line",

            x0=site_ratings_tv_median,

            y0=0,

            x1=site_ratings_tv_median,

            y1=170,

            line=dict(

                color="LightSeaGreen",

                width=4,

                dash="dashdot",

            ),

    )



fig.add_shape(

        # Line Horizontal

            type="line",

            x0=imdb_ratings_tv_median,

            y0=0,

            x1=imdb_ratings_tv_median,

            y1=170,

            line=dict(

                width=4,

                dash="dot",

            ),

    )

fig.update_layout(title="Distribution Plot Local Ratings Vs IMDB |TV")

# fig.write_image("../Imgs/Distribution_plt_lcl_ratings_tv.png")
site_ratings_tv = tv_df.sort_values('USER_REVIEWS_LOCAL', ascending=False,ignore_index=True)

site_ratings_tv.drop_duplicates('TV_NAME',inplace=True )

eval_tv_df= site_ratings_tv[['TV_NAME', 'IMDB','USER_REVIEWS_LOCAL','NUMBER_REVIEWS_LOCAL','SITE_RANK']]

eval_tv_df[['IMDB','USER_REVIEWS_LOCAL']] = scaler.fit_transform(eval_tv_df[['IMDB','USER_REVIEWS_LOCAL']])
tv_names = eval_tv_df.TV_NAME.values.tolist()

tv_imdb_ratings = eval_tv_df.IMDB.values.tolist()

tv_local_ratings = eval_tv_df.USER_REVIEWS_LOCAL.values.tolist()

fig = go.Figure()

fig.add_trace(go.Bar(x=tv_names[:100], y= tv_imdb_ratings[:100], name="IMDB Ratings Top 100|TV"))

fig.add_trace(go.Bar(x=tv_names[:100], y= tv_local_ratings[:100], name="Local Ratings Top 100|TV"))

fig.update_xaxes(tickangle=-45)

fig.update_layout(xaxis_title ="TV Shows", yaxis_title="Ratings", title="IMDB Vs Local Ratings Top 100,|TV |sort Local Ratings", height=700,)
fig = go.Figure()

fig.add_trace(go.Bar(x=tv_names[-100:], y= tv_imdb_ratings[-100:], name="IMDB Ratings Bottom 100|TV"))

fig.add_trace(go.Bar(x=tv_names[-100:], y= tv_local_ratings[-100:], name="Local Ratings Bottom 100|TV"))

fig.update_xaxes(tickangle=-45)

fig.update_layout(xaxis_title ="TV Shows", yaxis_title="Ratings", title="IMDB Vs Local Ratings Bottom 100,|TV |sort Local Ratings", height=700,)
imdb_ratings_tv= tv_df.sort_values('IMDB', ascending=False, ignore_index=True)

imdb_ratings_tv.drop_duplicates('TV_NAME',inplace=True )
eval_tv_df_imdb = imdb_ratings_tv[['TV_NAME', 'IMDB','USER_REVIEWS_LOCAL','NUMBER_REVIEWS_LOCAL','SITE_RANK']]

eval_tv_df_imdb[['IMDB','USER_REVIEWS_LOCAL']] = scaler.fit_transform(eval_tv_df_imdb[['IMDB','USER_REVIEWS_LOCAL']])
tv_names_imdb = eval_tv_df_imdb.TV_NAME.values.tolist()

tv__imdb_ratings_imdb = eval_tv_df_imdb.IMDB.values.tolist()

tv_local_ratings_imdb = eval_tv_df_imdb.USER_REVIEWS_LOCAL.values.tolist()
fig = go.Figure()

fig.add_trace(go.Bar(x=tv_names_imdb[:100], y= tv__imdb_ratings_imdb[:100], name="IMDB Ratings Top 100|TV"))

fig.add_trace(go.Bar(x=tv_names_imdb[:100], y= tv_local_ratings_imdb[:100], name="Local Ratings Top 100|TV"))

fig.update_xaxes(tickangle=-45)

fig.update_layout(xaxis_title ="TV Shows", yaxis_title="Ratings", title="IMDB Vs Local Ratings Top 100,|TV |sort IMDB Ratings", height=700,)
fig = go.Figure()

fig.add_trace(go.Bar(x=tv_names_imdb[-100:], y= tv__imdb_ratings_imdb[-100:], name="IMDB Ratings Bottom 100|TV"))

fig.add_trace(go.Bar(x=tv_names_imdb[-100:], y= tv_local_ratings_imdb[-100:], name="Local Ratings Bottom 100|TV"))

fig.update_xaxes(tickangle=-45)

fig.update_layout(xaxis_title ="TV Shows", yaxis_title="Ratings", title="IMDB Vs Local Ratings Bottom 100,|TV |sort IMDB Ratings", height=700,)
genre_prp_mv = movie_df.GENRE.value_counts(normalize=True)*100


genre_names_mv = genre_prp_mv.index.tolist()

genre_names_pct_mv = genre_prp_mv.values.tolist()



fig = go.Figure()

fig.add_trace(go.Pie(labels=genre_names_mv, values=genre_names_pct_mv))

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=15,

                  marker=dict( line=dict(color='#000000', width=2)),)

fig.update_layout(title="Genres In Movies",

                 template="ggplot2",

                    font=dict(

                        family="Courier New, monospace",

                              size=18,

                              color="#7f7f7f"))



# fig.write_image("../Imgs/genre_n_movies.png")
genre_prp_tv = tv_df.GENRE.value_counts(normalize=True)*100


genre_names_tv = genre_prp_tv.index.tolist()

genre_names_pct_tv = genre_prp_tv.values.tolist()
fig = go.Figure()

fig.add_trace(go.Pie(labels=genre_names_tv, values=genre_names_pct_tv))

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=15,

                  marker=dict( line=dict(color='#000000', width=2)),)

fig.update_layout(title="Genres In Movies",

                 template="ggplot2",

                    font=dict(

                        family="Courier New, monospace",

                              size=18,

                              color="#7f7f7f"))



# fig.write_image("../Imgs/genre_n_tvs.png")
fig = go.Figure()

fig.add_trace(go.Bar(x=genre_names_mv, y=genre_names_pct_mv,name="Genres| Movie"))

fig.add_trace(go.Bar(x=genre_names_tv, y=genre_names_pct_tv, name="GenresTV"))

fig.update_layout(title="Movie Vs. TV | Genre", xaxis_title="Genre Names", yaxis_title="Share in Percentage",template="ggplot2",)



# fig.write_image("../Imgs/movies_vs_tv_top_genre.png")
print(tv_df.drop_duplicates('TV_NAME').shape[0])

print(movie_df.drop_duplicates('MOVIE_NAME').shape[0])
movie_df.columns
site_ranks_movie = movie_df[['MOVIE_NAME','SITE_RANK','USER_REVIEWS_LOCAL','IMDB','NUMBER_REVIEWS_LOCAL']].sort_values('SITE_RANK')

site_ranks_movie.drop_duplicates('MOVIE_NAME', inplace=True)
site_ranks_movie.head()
# site_ranks_movie[['USER_REVIEWS_LOCAL','MOVIE_NAME','IMDB']].iloc[:50,:].iplot(kind='bar', x="MOVIE_NAME")

layout=dict(title="IMDB & 30 Most Watched Movies", 

            xaxis_title="Movies",

            yaxis_title="Ratings")

site_ranks_movie[['MOVIE_NAME','IMDB']].iloc[:30,:].iplot(kind="bar", x="MOVIE_NAME",layout=layout)

layout=dict(title="IMDB & 30 Least Watched Movies", 

            xaxis_title="Movies",

            yaxis_title="Ratings")

site_ranks_movie[['MOVIE_NAME','IMDB']].iloc[-30:,:].iplot(kind="bar", x="MOVIE_NAME",layout=layout)

movie_df[movie_df.MOVIE_NAME == "Eternal Sunshine Of The Spotless Mind"]['GENRE']
top_100_movies = site_ranks_movie.iloc[:100,0].values.tolist()

bottom_100_movies = site_ranks_movie.iloc[-100:,0].values.tolist()


top_100_mv_genres = np.array([])

bottom_100_mv_genres = np.array([])



for movie in top_100_movies:

    val = movie_df[movie_df.MOVIE_NAME == movie]['GENRE'].values

    top_100_mv_genres = np.append(top_100_mv_genres,val)

    

    

for movie in bottom_100_movies:

    val = movie_df[movie_df.MOVIE_NAME == movie]['GENRE'].values

    bottom_100_mv_genres = np.append(bottom_100_mv_genres,val)





top_100_mv_genre_counts = np.unique(top_100_mv_genres,return_counts=True, )

bottom_100_mv_genre_counts = np.unique(bottom_100_mv_genres,return_counts=True, )
top_100_mv_genre_counts[1]
fig = go.Figure()

fig.add_trace(go.Bar(x=top_100_mv_genre_counts[0], y=top_100_mv_genre_counts[1], name="Top 100 Most Watched Movies"))

fig.add_trace(go.Bar(x=bottom_100_mv_genre_counts[0], y=bottom_100_mv_genre_counts[1], name="Bottom 100 Most Watched Movies"))

fig.update_layout(title="Views & Genres|Movies", xaxis_title="Genres", yaxis_title="Counts")

# fig.write_image("../Imgs/views_n_genres_mv.png")
site_ranks_tv = tv_df[['TV_NAME','SITE_RANK','USER_REVIEWS_LOCAL','IMDB','NUMBER_REVIEWS_LOCAL']].sort_values('SITE_RANK')

site_ranks_tv.drop_duplicates('TV_NAME', inplace=True)

top_100_tvs = site_ranks_tv.iloc[:100,0].values.tolist()

bottom_100_tvs = site_ranks_tv.iloc[-100:,0].values.tolist()



top_100_tv_genres = np.array([])

bottom_100_tv_genres = np.array([])



for tv in top_100_tvs:

    val = tv_df[tv_df.TV_NAME == tv]['GENRE'].values

    top_100_tv_genres = np.append(top_100_tv_genres,val)

    

    

for tv in bottom_100_tvs:

    val = tv_df[tv_df.TV_NAME == tv]['GENRE'].values

    bottom_100_tv_genres = np.append(bottom_100_tv_genres,val)





top_100_tv_genre_counts = np.unique(top_100_tv_genres,return_counts=True, )

bottom_100_tv_genre_counts = np.unique(bottom_100_tv_genres,return_counts=True, )
fig = go.Figure()

fig.add_trace(go.Bar(x=top_100_tv_genre_counts[0], y=top_100_tv_genre_counts[1], name="Top 100 Most Watched TVs"))

fig.add_trace(go.Bar(x=bottom_100_tv_genre_counts[0], y=bottom_100_tv_genre_counts[1], name="Bottom 100 Most Watched TVs"))

fig.update_layout(title="Views & Genres|TV", xaxis_title="Genres", yaxis_title="Counts")

# fig.write_image("../Imgs/views_n_genres_tv.png")
temp_df = movie_df.drop_duplicates("MOVIE_NAME")
grp_by_released_date_movie = temp_df['MOVIE_NAME'].groupby(temp_df.index.year).count()
index = grp_by_released_date_movie.index.tolist()

values = grp_by_released_date_movie.values.tolist()

size = [val*.5 for val in values]
fig= go.Figure()

fig.add_trace(go.Scatter(x =index,y= values, mode="markers", marker=dict(size=size,color=size)))

fig.update_layout(title="Movies Released Per Year", xaxis_title="Year", yaxis_title="Movies Counts")

fig.show()

# grp_by_released_date_movie.iplot(kind="scatter", mode="markers", size=grp_by_released_date_movie.values*0.5)

# fig.write_image("../Imgs/Movie_releases_per_yr.png")
yr_most_watched_rank  = temp_df.groupby(temp_df.index.year)[["SITE_RANK"]].mean().sort_values('SITE_RANK')
rank = yr_most_watched_rank.rank(method='min')



x = rank.index.tolist()

y =  rank.SITE_RANK.values





#Lets subtract all values by 60 as max value being 42. By doing so smallest rank becomes highest and gets biggest size in plot. 

rank_plot_size=[60-float(val) for val in y]
fig= go.Figure()

fig.add_trace(go.Scatter(x=x,y=y, mode='markers', marker=dict(size=rank_plot_size,color=rank_plot_size)))

fig.update_layout(title="Rank of Years By Most Watched|Movie",xaxis_title="Year", yaxis_title="Rank", )

fig.show()



# fig.write_image("../Imgs/yr_rank_most_watched_mv.png")