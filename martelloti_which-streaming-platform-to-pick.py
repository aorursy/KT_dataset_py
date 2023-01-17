import numpy as np

import pandas as pd

import plotly.graph_objects as go

import time

import seaborn as sns

import matplotlib.pyplot as plt
movies_df = pd.read_csv('/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv')



movies_df.info()
movies_df.head()
movies_df = pd.melt(movies_df, id_vars = ['ID', 'Title', 'Year', 'Age', 'IMDb', 'Rotten Tomatoes', 'Type', 'Directors', 'Genres', 'Country', 'Language'], 

        value_vars = ['Netflix', 'Hulu', 'Prime Video', 'Disney+'], var_name = 'Streaming Platform')

movies_df = movies_df[movies_df['value'] == 1].drop(['value', 'Type'], axis = 1) #Type is being dropped here because the value will always be 0, since the whole dataset represents only movies

movies_df.head()
movies_count = movies_df.groupby('Streaming Platform', as_index = False).count()

movies_count = movies_count[['Streaming Platform', 'ID']].rename({'ID' : 'Count'}, axis = 'columns')

movies_count = movies_count.sort_values(by = 'Count', ascending = False)

color_map={

    "Prime Video": "rgb(4, 165, 222)",

    "Netflix": "rgb(223, 9, 18)",

    "Hulu": "rgb(2, 228, 119)",

    "Disney+": "rgb(0, 0, 0)"

}



import plotly.express as px



fig = px.bar(movies_count, y='Streaming Platform', x="Count", color="Streaming Platform", orientation="h",

             color_discrete_map= color_map, text = "Count"

            )



fig.update_layout(

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)',

    title="Number of movies offered by platform",

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Streaming Platform',

        titlefont_size=16,

        tickfont_size=14

    ),

    legend=dict(

        x=1,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

)



fig.show()
list_of_sp = movies_count['Streaming Platform'].tolist()



movies_df_not_null = movies_df[~movies_df['IMDb'].isnull()]



fig = go.Figure()



rows_counter = 0

for sp, clr in zip(list_of_sp, color_map.values()):

        

        fig.add_trace(go.Box(

            x=movies_df_not_null[movies_df_not_null['Streaming Platform'] == sp]['Streaming Platform'],

            y=movies_df_not_null[movies_df_not_null['Streaming Platform'] == sp]['IMDb'],

            name=sp,

            boxpoints='all',

            jitter=0.5,

            whiskerwidth=0.2,

            fillcolor=clr,

            marker_size=2,

            line_width=1)

        )

        

        min_value = round(np.percentile(movies_df_not_null[movies_df_not_null['Streaming Platform'] == sp]['IMDb'], 0), 2)

        first_quartile = round(np.percentile(movies_df_not_null[movies_df_not_null['Streaming Platform'] == sp]['IMDb'], 25), 2)

        median = round(np.percentile(movies_df_not_null[movies_df_not_null['Streaming Platform'] == sp]['IMDb'], 50), 2)

        third_quartile = round(np.percentile(movies_df_not_null[movies_df_not_null['Streaming Platform'] == sp]['IMDb'], 75), 2)

        max_value = round(np.percentile(movies_df_not_null[movies_df_not_null['Streaming Platform'] == sp]['IMDb'], 100), 2)

        

        for y_desc, y_value in zip(('MinV: ', 'Q1: ', 'Med: ', 'Q3: ', 'MaxV: '), (min_value, first_quartile, median, third_quartile, max_value)):

        

            fig.add_annotation(

                        x=rows_counter + 0.35,

                        ax = 0,

                        ay = 0,

                        y=y_value,

                        text=y_desc + str(y_value))



        rows_counter += 1

        

fig.update_layout(

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)',

    title='Distribution of movies IMDb score by streaming platform',

    yaxis=dict(

        title='IMDb score',

        titlefont_size=16,

        tickfont_size=14

    ))



        

fig.show()
movies_df.sort_values(by = 'IMDb', ascending = False, inplace = True)

results_list = {}

for sp in movies_count['Streaming Platform'].tolist():

    sp_best_df = movies_df[movies_df['Streaming Platform'] == sp].head(n = 52)

    IMDb_mean = round(sp_best_df['IMDb'].mean(), 2)

    results_list[sp] = IMDb_mean

best_movies_df = pd.DataFrame.from_dict(results_list, orient = 'index', columns = ['AVG Score']).sort_values(by = 'AVG Score', ascending = True)
fig = px.bar(best_movies_df, y=best_movies_df.index, x="AVG Score", color=best_movies_df.index, orientation="h",

             color_discrete_map = color_map, text = "AVG Score"

            )



fig.update_layout(

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)',

    title="Average score of the best 52 movies by platform",

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Streaming Platform',

        titlefont_size=16,

        tickfont_size=14

    ),

    xaxis = dict(

        range = [best_movies_df['AVG Score'].min() - 0.1, best_movies_df['AVG Score'].max() + 0.1],

        title = "Average Score"

    ),

    legend=dict(

        x=1,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)',

        title = 'Streaming Platform'

    ),

)



fig.show()
genres_split = movies_df['Genres'].str.get_dummies(',')

genres_movie_df = pd.concat([movies_df, genres_split], axis = 1)

genres_movie_df = pd.melt(genres_movie_df, id_vars = ['ID', 'Title', 'Year', 'Age', 'IMDb', 'Rotten Tomatoes', 'Directors', 'Genres', 'Country', 'Language', 'Streaming Platform'], 

        value_vars = genres_split.columns, var_name = 'Genre')

genres_movie_df = genres_movie_df[genres_movie_df['value'] == 1].drop(['value', 'Genres'], axis = 1)

genres_movie_df.head()
genres_count = genres_movie_df.groupby('Genre', as_index = False).count()

genres_count = genres_count[['Genre', 'ID']].rename({'ID' : 'Count'}, axis = 'columns')

genres_count = genres_count.sort_values(by = 'Count', ascending = False)



import plotly.express as px



fig = px.bar(genres_count.head(n = 15), y='Genre', x="Count", color='Genre', orientation="h", text = "Count"

            )



fig.update_layout(

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)',

    title="Number of movies segmented by genre",

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Movie genre',

        titlefont_size=16,

        tickfont_size=14

    ),

    legend=dict(

        x=1,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

)



fig.show()
def segmented_boxplot_function(df, df_count, segmentation_column, segmentation_number):

   

    df_not_null = df[~df['IMDb'].isnull()]

    list_top_segments = df_count[segmentation_column].tolist()

    segment = list_top_segments[segmentation_number]

    fig = go.Figure()



    rows_counter = 0

    for sp, clr in zip(list_of_sp, color_map.values()):



            query_filter = (df_not_null['Streaming Platform'] == sp) & (df_not_null[segmentation_column] == segment)



            fig.add_trace(go.Box(

                x=df_not_null[query_filter]['Streaming Platform'],

                y=df_not_null[query_filter]['IMDb'],

                name=sp,

                boxpoints='all',

                jitter=0.5,

                whiskerwidth=0.2,

                fillcolor=clr,

                marker_size=2,

                line_width=1)

            )



            min_value = round(np.percentile(df_not_null[query_filter]['IMDb'], 0), 2)

            first_quartile = round(np.percentile(df_not_null[query_filter]['IMDb'], 25), 2)

            median = round(np.percentile(df_not_null[query_filter]['IMDb'], 50), 2)

            third_quartile = round(np.percentile(df_not_null[query_filter]['IMDb'], 75), 2)

            max_value = round(np.percentile(df_not_null[query_filter]['IMDb'], 100), 2)



            for y_desc, y_value in zip(('MinV: ', 'Q1: ', 'Med: ', 'Q3: ', 'MaxV: '), (min_value, first_quartile, median, third_quartile, max_value)):



                fig.add_annotation(

                            x=rows_counter + 0.35,

                            ax = 0,

                            ay = 0,

                            y=y_value,

                            text=y_desc + str(y_value))



            rows_counter += 1



    fig.update_layout(

                paper_bgcolor='rgba(0,0,0,0)',

                plot_bgcolor='rgba(0,0,0,0)',

                title='Distribution of movies IMDb score by ' + segment,

                yaxis=dict(

                    title='IMDb score',

                    titlefont_size=16,

                    tickfont_size=14

                ))





    fig.show()

    
segmented_boxplot_function(genres_movie_df, genres_count, 'Genre', 0)
segmented_boxplot_function(genres_movie_df, genres_count, 'Genre', 1)
segmented_boxplot_function(genres_movie_df, genres_count, 'Genre', 2)
segmented_boxplot_function(genres_movie_df, genres_count, 'Genre', 3)
segmented_boxplot_function(genres_movie_df, genres_count, 'Genre', 4)
countries_split = movies_df['Country'].str.get_dummies(',')

countries_movie_df = pd.concat([movies_df, countries_split], axis = 1)

countries_movie_df = pd.melt(countries_movie_df, id_vars = ['ID', 'Title', 'Year', 'Age', 'IMDb', 'Rotten Tomatoes', 'Directors', 'Genres', 'Language', 'Streaming Platform'], 

        value_vars = countries_split.columns, var_name = 'Country')

countries_movie_df = countries_movie_df[countries_movie_df['value'] == 1].drop(['value'], axis = 1)

countries_movie_df
countries_count = countries_movie_df.groupby('Country', as_index = False).count()

countries_count = countries_count[['Country', 'ID']].rename({'ID' : 'Count'}, axis = 'columns')

countries_count = countries_count.sort_values(by = 'Count', ascending = False)



import plotly.express as px



fig = px.bar(countries_count.head(n = 15), y='Country', x="Count", color='Country', orientation="h", text = "Count"

            )



fig.update_layout(

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)',

    title="Number of movies segmented by genre",

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Movie genre',

        titlefont_size=16,

        tickfont_size=14

    ),

    legend=dict(

        x=1,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

)



fig.show()
segmented_boxplot_function(countries_movie_df, countries_count, 'Country', 0)
segmented_boxplot_function(countries_movie_df, countries_count, 'Country', 1)
segmented_boxplot_function(countries_movie_df, countries_count, 'Country', 2)
segmented_boxplot_function(countries_movie_df, countries_count, 'Country', 3)