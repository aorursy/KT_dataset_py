import pandas as pd # pandas is a data manipulation library
import numpy as np #provides numerical arrays and functions to manipulate the arrays efficiently
import random
import matplotlib.pyplot as plt # data visualization library
from wordcloud import WordCloud, STOPWORDS #used to generate world cloud
from pandas import DataFrame as df
#importing movies.csv
movies_df= pd.read_csv('../input/movies.csv')
#dimensionality of movies df
movies_df.shape
#top 5 rows of movies_df
movies_df.head()
#a concise summary of 
movies_df.info()
len(movies_df.index)
# lets explore ratings.CSV
ratings_df=pd.read_csv('../input/ratings.csv',sep=',')
ratings_df.shape
ratings_df.head()
#summary of ratings.csv
ratings_df.describe()
ratings_df.info()
#is any row null
movies_df.isnull().any()
#is any row null there
ratings_df.isnull().any()
#spliting genres
movies_df['genres_arr'] = movies_df['genres'].str.split('|')
movies_df.head()
del movies_df['genres']
movies_df.head()
#count how many genres a movie have
counter_lambda = lambda x: len(x)
movies_df['genre_count'] = movies_df.genres_arr.apply(counter_lambda)
movies_df.head()
#count how many movies are of Genre of Animation
animation_df = movies_df[movies_df.genres_arr.map(lambda x: 'Animation' in x)]
print (len(animation_df.index))
from collections import Counter

flattened_genres = [item for sublist in movies_df.genres_arr for item in sublist]
genre_dict = dict(Counter(flattened_genres))
import pprint
pprint.pprint (genre_dict)
#minimum rating given to a movie
min_rate=ratings_df['rating'].min()
min_rate
min_rate_df=ratings_df.loc[ratings_df['rating'] == min_rate]
min_rate_df.head(3)
#maximum rating given to a movie
max_rate=ratings_df['rating'].max()
max_rate
max_rate_df=ratings_df.loc[ratings_df['rating'] == max_rate]
max_rate_df.head(3)
# lets find average rating using the numpy's mean method
np.mean(ratings_df.rating)
# lets find the median rating
np.median(ratings_df.rating)
# lets find the 30th percentile rating 
np.percentile(ratings_df.rating, 30)
# lets find the most common rating given by users to movies (called mode of the data)
from scipy import stats
stats.mode(ratings_df.rating)
# now lets plot a histogram of movie ratings to get an overall picture
plt.hist(ratings_df.rating)
plt.xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
plt.xlabel('Rating')
plt.ylabel('# of movies')
plt.grid()
plt.show()
#merging two dataframes "movies.csv" and "ratings.csv"
movie_df_ratings_df=movies_df.merge(ratings_df,on = 'movieId',how = 'inner')
movie_df_ratings_df.head(3)
len(movie_df_ratings_df)
# now lets find the titles of the top 10 movies to see if we are missing on some awesome movies!
titles_df = movie_df_ratings_df[['movieId', 'title', 'rating']]
titles_df.groupby(['movieId', 'title'], as_index=False).mean().sort_values(by='rating', ascending=False).head(10)
# these movies are not what we expected to be in the Top 10 movies list, something's wrong here
# lets check how many ratings have these movies received, lets take an example of movieId 163949
len(movie_df_ratings_df[movie_df_ratings_df['movieId'] == 1706].index) # 163949
# now lets only consider movies which have atleast 100 ratings and see how the top 10 movies change
temp_df = titles_df.groupby(['movieId', 'title'], as_index=False).count()
well_rated_df = temp_df[temp_df['rating'] > 100].sort_values(by='rating', ascending=False)
well_rated_df.head()
# now lets created a filtered df from merged_df which only has these movies and then find top 20 movies
filtered_df = movie_df_ratings_df[movie_df_ratings_df['movieId'].apply(lambda x: x in list(well_rated_df['movieId']))]
titles_df = filtered_df[['title', 'rating', 'movieId']]
titles_df.groupby(['movieId', 'title'], as_index=False).mean().sort_values(by='rating', ascending=False).head(20)
#displays high rated movies
high_rated= movie_df_ratings_df['rating']>4.0
movie_df_ratings_df[high_rated].head(10)
# displays low rated movies
low_rated = movie_df_ratings_df['rating']<1.0
movie_df_ratings_df[low_rated].head()
#top 25 most rated movies
most_rated = movie_df_ratings_df.groupby('title').size().sort_values(ascending=False)[:25]
most_rated.head(25)
#slicing out columns to display only title and genres columns from movies.csv
movies_df[['title','genres_arr']].head()
# now lets merge/join the movies_df and ratings_df so that we can see the actual movie titles of top 10 movies
merged_df = pd.merge(ratings_df, movies_df, on='movieId')
merged_df.head()
# now lets add a column called rating_year which depicts the year when the rating was given
import datetime
year_lambda = lambda x: int(datetime.datetime.fromtimestamp(x).strftime('%Y'))
merged_df['rating_year'] = merged_df['timestamp'].apply(year_lambda)
merged_df.head()
merged_df.to_csv("merged.csv",sep=',', encoding='utf-8')
# now lets create a new data frame which contains number of ratings given on each year
ratings_per_year = merged_df.groupby(['rating_year'])['rating_year'].count()
ratings_per_year.head(5)

# now lets create a new data frame which contains number of ratings given on each year
ratings_num = merged_df.groupby(['rating_year'])['rating'].count()
ratings_num.head(5)
# now lets get some stats on number of ratings per year
years = ratings_num.keys()
num_ratings = ratings_num.get_values()
print ('average ratings per year', np.mean(num_ratings))
print ('median ratings per year', np.median(num_ratings))
print ('90% ratings per year', np.percentile(num_ratings, 90))
# now lets scatter plot this data to visualize how ratings are spead across years
plt.scatter(years, num_ratings)
plt.title('# of rating across years')
plt.xlabel('Year')
plt.ylabel('# of ratings')
plt.show()
# now lets plot this genre distribution as a pie chart
plt.pie(genre_dict.values(), labels=genre_dict.keys())
plt.title('Genre distribution of movies')
plt.savefig('./movie-genres-pie.png')
plt.show()
# Function that control the color of the words
def random_color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None):
    h = int(360.0 * tone / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)
tone = 100

f, ax = plt.subplots(figsize=(14, 6))

wordcloud = WordCloud(width=550,height=300, background_color='black', 
                      max_words=1628,relative_scaling=0.7,
                      color_func = random_color_func,
                      normalize_plurals=False)

wordcloud.generate_from_frequencies(genre_dict)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.savefig('./wordcloud.png')
plt.show()
# we can also plot a bar chart (with grid lines and slanted x axis labels for better readability)
x = list(range(len(genre_dict)))
plt.xticks(x, genre_dict.keys(), rotation=80)
plt.bar(x, genre_dict.values())
plt.grid()
plt.show()
# now lets try to build a linear regression model using which we will predict how many ratings we get each year
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(years, num_ratings)
print ('Generated linear model is  y = ' + str(slope) + ' * x + ' + str(intercept))
# now lets use the slope and intercept to create a predict function which will predict num_ratings given a year
def predict_num_ratings(year):
    return slope * year + intercept

predicted_ratings = predict_num_ratings(years)
# now lets plot our predicted values along side the actual data to see how well we did
plt.scatter(years, num_ratings)
plt.plot(years, predicted_ratings, c='r')
plt.show()
# now lets see how good our prediction is by calculating the r-squared value
r_square = r_value ** 2
print ('Linear Model r_square value', r_square)
# now lets try a polynomial function instead of a linear function and see if that fits better
polynomial = np.poly1d(np.polyfit(years, num_ratings, 3))
plt.scatter(years, num_ratings)
plt.plot(years, polynomial(years), c='r')
plt.show()
# now lets calculate the r-square for this polynomial regression

from sklearn.metrics import r2_score
r2 = r2_score(num_ratings, polynomial(years))
print ('Polynomial Model r_square value', r2)
# now we can predict how many ratings we expect in any year using our polynomial function
print(predict_num_ratings(2030))
# now lets try to build a linear regression model using which we will predict how many ratings we get each year
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(years, num_ratings)
print ('Generated linear model is  y = ' + str(slope) + ' * x + ' + str(intercept))
# now lets use the slope and intercept to create a predict function which will predict num_ratings given a year
def predict_num_ratings(year):
    return slope * year + intercept

predicted_ratings = predict_num_ratings(years)
# now lets plot our predicted values along side the actual data to see how well we did
plt.scatter(years, num_ratings)
plt.plot(years, predicted_ratings, c='r')
plt.show()
# now lets see how good our prediction is by calculating the r-squared value
r_square = r_value ** 2
print ('Linear Model r_square value', r_square)
# now lets try a polynomial function instead of a linear function and see if that fits better
polynomial = np.poly1d(np.polyfit(years, num_ratings, 3))
plt.scatter(years, num_ratings)
plt.plot(years, polynomial(years), c='r')
plt.show()
# now lets calculate the r-square for this polynomial regression

from sklearn.metrics import r2_score
r2 = r2_score(num_ratings, polynomial(years))
print ('Polynomial Model r_square value', r2)
# now we can predict how many ratings we expect in any year using our polynomial function
print(predict_num_ratings(2030))
# now lets try a polynomial function to predict reviews in numbers
poly= np.poly1d(np.polyfit(num_ratings,ratings, 5))
# now we can predict how many ratings we expect in numbers using our polynomial function
print (poly(2100))              