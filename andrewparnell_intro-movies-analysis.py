# Import Pandas, the Python library built for analyzing data

import pandas

# Load up my csv of movie ratings and convert it into a pandas dataframe object

movies = pandas.read_csv('/kaggle/input/andrews-movies/ratings.csv')
# Display the "head" or the top 5 rows of data to get a feel for what the data looks like, column names, etc

movies.head()
# Now lets do something cool, liiiike how about see my average rating by film release year?

# First group the rows by the Year column, average all the rows that get grouped, and show me the Rating column

movies.groupby('Year').mean()['Rating']
# Hmm, thats pretty good, but why dont we sort it by rating and then look at the top 10 so I can see which years were the best

ratings_by_year = movies.groupby('Year').mean()['Rating']

ratings_by_year.sort_values(ascending=False)[0:10]
# 1981? Weird, what movies came out in 1981?

movies[movies['Year'] == 1981]
# Ah, Guess that will do it - how many 5 star movies do I have anyway??

movies[movies['Rating'] == 5]['Rating'].count()
# Cool, and what are they?

movies[movies['Rating'] == 5].sort_values(['Year'])
# I wonder what a scatter plot of movies might look like for this

movies.plot.scatter('Year','Rating',figsize=(20,8))