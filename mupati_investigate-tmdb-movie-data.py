# import the necessary packages

import pandas as pd

import numpy as np

import matplotlib.pyplot  as plt

import seaborn as sns



# Initialize seaborn and magic commands

sns.set()

%matplotlib inline
# load tmdb-movies csv into a dataframe

movies_df = pd.read_csv('../input/tmdb-movies-dataset/tmdb-movies.csv')



# Examine first 5 rows

movies_df.head(5)
# check the number of rows and columns

movies_df.shape
# Get a quick statistics of the dataset

movies_df.describe()
# View more details about the dataset

movies_df.info()
# check the columns with empty values

movies_df.isnull().sum()
# check for number of duplicates

movies_df.duplicated().sum()
# remove duplicates

movies_df.drop_duplicates(inplace=True)
# verify duplicates removal

movies_df.duplicated().sum()
# confirm the new number of rows

movies_df.shape
# drop imdb_id, homepage, tagline and overview columns. 

movies_df.drop(['imdb_id', 'homepage','tagline', 'overview'], axis=1, inplace=True )

movies_df.shape
# remove rows if the budget or revenue or budget_adj or revenue_adj is 0.0

movies_df = movies_df.loc[(movies_df.budget * movies_df.revenue * movies_df.budget_adj* movies_df.revenue_adj) != 0]

movies_df.shape
# fill the missing values in the following columns with the values indicated

# cast = 'no_cast'

# director = 'no_director'

# keywords = 'no_keywords'

# genres = 'no_genres'

# production_companies = 'no_production_companies'



columns_to_fillna = ['cast', 'director', 'keywords', 'genres', 'production_companies']

for column in columns_to_fillna:

    movies_df[column] = movies_df[column].fillna('no_'+column)



# check the number of null values again

movies_df.isnull().sum()
def generate_plot(x_val, y_val, fig_size, title, x_label, y_label):

    """

    This functions takes inputs for a bar graph and produces a plot based on the inputs

    """

    plt.subplots(figsize=fig_size)

    sns.barplot(x_val, y_val)

    plt.title(title, fontsize=30)

    plt.xlabel(x_label, fontsize=20)

    plt.ylabel(y_label, fontsize=20);
def generate_value_and_count(data):

    """

    This functions takes a column and separates the pipe-separated values and return a dict of

    the value and the number of times it occurs

    """    

    val_list = [val.split('|') for val in data]

    

    top_val_list = []

    for new_val in val_list:

        for single in new_val:

            top_val_list.append(single)



    # get the value and count of each item in the top_val_list

    val_and_count = dict()

    for i in top_val_list:

        val_and_count[i] = val_and_count.get(i, 0)+1

    

    return val_and_count
# Get the Most Popular Genre in Each Release Year

popular_genre_per_year = movies_df.groupby(['release_year'])[['vote_average','genres']].max()

plt.subplots(figsize=(25, 20))

graph = sns.barplot(

                    popular_genre_per_year.index,

                    popular_genre_per_year['vote_average'],

                    hue=popular_genre_per_year['genres'],

                    dodge=False,

                    palette='muted',

                   )

graph.set_xticklabels(graph.get_xticklabels(),

                      rotation=90,

                      fontweight='light',fontsize='xx-large'

                     )

graph.axes.set_title("The Yearly Most Popular Movie Genre From 1960 to 2015",fontsize=40)

graph.set_xlabel("Genres",fontsize=30)

graph.set_ylabel("Popularity",fontsize=30);



# Put the legend out of the figure

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.1, prop={'size': 20});

# Find the most popular genre in the first earliest decade

earliest_decade = popular_genre_per_year.head(10)['genres']

earliest_decade_genres = generate_value_and_count(data=earliest_decade)

earliest_decade_genres_df = pd.DataFrame.from_dict(earliest_decade_genres, orient="index")

# Generate the Plot

generate_plot(x_val=earliest_decade_genres_df[0],

              y_val=earliest_decade_genres_df.index,

              fig_size=(20,10),

              title='Most Popular Genres In Earliest Decade',

              x_label='Number of Movies',

              y_label='Genre'

             )
# Find the most popular genre in the most recent decade

recent_decade = popular_genre_per_year.tail(10)['genres']

recent_decade_genres = generate_value_and_count(data=recent_decade)

recent_decade_genres_df = pd.DataFrame.from_dict(recent_decade_genres, orient="index")

# Generate the Plot

generate_plot(x_val=recent_decade_genres_df[0],

              y_val=recent_decade_genres_df.index,

              fig_size=(20,10),

              title='Most Popular Genres In Recent Decade',

              x_label='Number of Movies',

              y_label='Genre'

             )
# Group The Genres by Year and Find the Most Popular Ones in Each Year using vote_average as the popularity metric

for year in np.arange(1960,2016, 10): # Interval can be changed if all the years need to appear.

    df = movies_df.query('release_year == @year').groupby('genres').mean().sort_values(by=['vote_average'], ascending=False).head(5)

    generate_plot(x_val=df.index,

                  y_val=df['vote_average'],

                  fig_size=(20,5),

                  title=f"5 Most Popular Genres Each {year}",

                  x_label=year,

                  y_label='Popularity'

                 )

# Find the Sum of the Various Columns According to the Year the Movies Were Released

yearly_movies_sum = movies_df.groupby('release_year').sum()



# Find The Yearly Change in Revenue

yearly_movies_sum.apply(lambda x:x.diff().fillna(0))[['revenue']].plot()

plt.xlabel('Year Released')

plt.ylabel('Change in Revenue(USD)')

plt.title('Yearly Change In Revenue');
# Compare the Yearly sum for budget_adj and revenue_adj

yearly_movies_sum[['budget_adj', 'revenue_adj']].plot()

plt.xlabel('Year Released')

plt.ylabel('Revenue & Budget (USD)')

plt.title('Revenue By Release Year');
# Compare the Yearly sum for budget and revenue

yearly_movies_sum[['budget', 'revenue']].plot()

plt.xlabel('Year Released')

plt.ylabel('Revenue & Budget (USD)')

plt.title('Revenue By Release Year');
# I chose movies with a Revenue Greater than the 90th Percentile as High Revenue Movies



# Calculate the 90th Percentile Revenue

ninety_percentile = np.percentile(movies_df['revenue_adj'], 90)



# Filter the movies with movies with revenue greater than 90th Percentile

highest_revenue_movies = movies_df.query('revenue_adj > @ninety_percentile')
# Check the Budget for of High Revenue Movies

highest_revenue_movies['budget_adj'].hist()

plt.xlabel('Budget (USD)')

plt.ylabel('Number of Movies')

plt.title('Number of Movies By Budget');
# Check the Vote Average (Popularity Metric) of High Revenue Movies

highest_revenue_movies['vote_average'].hist()

plt.xlabel('Average Votes')

plt.ylabel('Number of Movies')

plt.title('Number of Movies By Average Vote');
# Check the Runtime of High Revenue Movies - How Long The Movies span

highest_revenue_movies['runtime'].hist()

plt.xlabel('Runtime')

plt.ylabel('Number of Movies')

plt.title('Number of Movies By Runtime');
# Check the Release Year of High Revenue Movies

highest_revenue_movies['release_year'].hist()

plt.xlabel('Year of Release')

plt.ylabel('Number of Movies')

plt.title('Number of Movies By Year of Release');

# Find the top 20 revenue movies

top_20_revenue = highest_revenue_movies.sort_values(by=['revenue_adj'], ascending=False).head(20)
# Find the Director That Featured on Most of the Highest Revenue Movies



director_counts = generate_value_and_count(data=top_20_revenue['director'])

director_count_df = pd.DataFrame.from_dict(director_counts, orient="index")



# Generate the Plot

generate_plot(x_val=director_count_df[0],

              y_val=director_count_df.index,

              fig_size=(20,10),

              title='Number of Top Movies Per Director',

              x_label='Number Of Top Movies',

              y_label='Director'

             )
# Find the Actor That Featured on Most of the Highest Revenue Movies

actor_counts = generate_value_and_count(data=top_20_revenue['cast'])

actor_count_df = pd.DataFrame.from_dict(actor_counts, orient="index").sort_values(by=[0], ascending=False).head(20)



# Generate the Plot

generate_plot(x_val=actor_count_df[0],

              y_val=actor_count_df.index,

              fig_size=(15,10),

              title='Top Movies Per Actor',

              x_label='Number Of Top Movies',

              y_label='Actors'

             )
# Find the Production Company That Featured in Most of the Highest Revenue Movies

company_counts = generate_value_and_count(data=top_20_revenue['production_companies'])

company_count_df = pd.DataFrame.from_dict(company_counts, orient="index")



# Generate the Plot

generate_plot(x_val=company_count_df[0],

              y_val=company_count_df.index,

              fig_size=(10,10),

              title='Top Movies Per Production Company',

              x_label='Number Of Top Movies',

              y_label='Production Company'

             )
# Generate plots for highest revenue movies

generate_plot(x_val=top_20_revenue['revenue_adj'],

              y_val=top_20_revenue['original_title'],

              fig_size=(10,10),

              title='Top Revenue Movies From 1960 to 2015',

              x_label='Revenue(USD)',

              y_label='Movie Title'

             )