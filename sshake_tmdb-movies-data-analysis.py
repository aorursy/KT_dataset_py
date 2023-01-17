# Reading the csv file into a pandas dataframe
import pandas as pd
import numpy as np

np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# I created two data frame objects one that contains the orginal data set, and another that contain and edit or cleaning done
tmdb_data = pd.read_csv('../input/tmdb-movies.csv')
tmdb_data_edited = pd.read_csv('../input/tmdb-movies.csv')

tmdb_data.head()
tmdb_data_edited.rename(columns = {'vote_average':'average_vote_score'}, inplace=True)
# Check to see if vote_average column was changed
tmdb_data_edited.columns
# Created this object to be called every time I need to check if a column being analyzed has any NaN values
tmdb_columns_with_nulls = tmdb_data_edited.isnull().any()
# This is a demonstration that shows the following line of code returns a dataframe containin rows for every NaN occurrence, 
#it will return a row  more than once if it contains a zero in more than once column
tmdb_data_edited[tmdb_data_edited.isnull().values].tail()
# The following creates and instantiates an object with the same dataframe from the previous cell with the dupicate rows removed
tmdb_nan_values_df = tmdb_data_edited[tmdb_data_edited.isnull().values].drop_duplicates()
tmdb_nan_values_df.tail() # This demonstrates that the duplicate rows have been removed
# Created this object to be called every to check if a column being analyzed has a zero values
tmdb_columns_with_zeros = (tmdb_data_edited == 0).any()
# As with cell containing code that returned the corresponding row for every instance of a NaN value, the line of code in this 
# cell demonstrates the dataframe containing corresponding row values for each instance of a zero value 
tmdb_data_edited[tmdb_data_edited.values == 0].head()
tmdb_zero_values_df = tmdb_data_edited[tmdb_data_edited.values == 0].drop_duplicates()
tmdb_zero_values_df.head()
print ((tmdb_data_edited == 0).any())
tmdb_columns_with_nulls
release_year_array = np.unique(tmdb_data_edited['release_year'].values)
print (len(release_year_array))
np.sort(release_year_array)
# Funtion to return a dataframe containing all the rows that carry the max per year for a particular column
def create_df_max_by_release_year(tmdb_dataframe, column_name_string):
    max_movies_per_year = tmdb_data_edited[tmdb_data_edited.groupby(['release_year'])[column_name_string].transform(max) == tmdb_data_edited[column_name_string]]
    return max_movies_per_year.sort_values(by=['release_year'])
# create_df_max_by_release_year is being used to return a dataframe containg all the movies that had the highest budget for their
# respective year
movies_with_max_adj_budget = create_df_max_by_release_year(tmdb_data_edited, 'budget_adj')
len(movies_with_max_adj_budget)
duplicate_max_budgets = movies_with_max_adj_budget[movies_with_max_adj_budget['release_year'].duplicated() == True]
print ('duplicate_max_budgets has ' + str(len(duplicate_max_budgets)) + ' items')
duplicate_max_budgets
from random import randint

movies_with_max_adj_budget[movies_with_max_adj_budget['release_year'].values == duplicate_max_budgets.iloc[randint(0,len(duplicate_max_budgets)-1)]['release_year']]
# As was done in an earlier cell to obtain movies_with_max_adj_budget, a similar method will now be used to obtain the highest
# revenue grossing movies pre year, and highest voted movies per year

# Max revenues
movies_with_max_adj_revenue = create_df_max_by_release_year(tmdb_data_edited, 'revenue_adj')
len(movies_with_max_adj_revenue)
# Max vote scores
movies_with_max_adj_revenue = create_df_max_by_release_year(tmdb_data_edited, 'revenue_adj')
len(movies_with_max_adj_revenue)
movies_with_max_average_vote_score = create_df_max_by_release_year(tmdb_data_edited, 'average_vote_score')
len(movies_with_max_average_vote_score)
movies_with_max_average_vote_score.head()
%matplotlib inline

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
def plot_tmdb_data_by_release_year(maxes_dataframe, column_to_plot_string, label_display_string=''):
    return plt.plot(maxes_dataframe['release_year'].values, maxes_dataframe[column_to_plot_string].values, label=label_display_string)
print (tmdb_data_edited['release_year'].min())
print (tmdb_data_edited['release_year'].max())

x_axis_numbering = np.linspace(tmdb_data_edited['release_year'].min(), tmdb_data_edited['release_year'].max(), 12)
x_axis_numbering
plt.figure(figsize=(25,10))

plt.xticks(x_axis_numbering)

largest_budget_revenues = plot_tmdb_data_by_release_year(movies_with_max_adj_budget, 'revenue_adj', 'Largest Budget Revenues')
highest_grossing_movies_revenues = plot_tmdb_data_by_release_year(movies_with_max_adj_revenue, 'revenue_adj', 'Highest Grossing Movies Revenues')

plt.title('Revenues per Year for Movie in max Category', fontsize = 20)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Revenue Generated', fontsize=20)

plt.legend()
plt.figure(figsize=(25,10))

plt.xticks(x_axis_numbering)

plot_tmdb_data_by_release_year(movies_with_max_adj_budget, 'average_vote_score', 'Largest Budget Avg Vote Scores')
plot_tmdb_data_by_release_year(movies_with_max_average_vote_score, 'average_vote_score', 'Highest Avg Vote Scores')

plt.title('Average Vote Score per Year for Movie in max Category', fontsize = 20)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Average Vote Score', fontsize=20)

plt.legend()
genre_grid = tmdb_data_edited['genres'].str.split('|',expand=True)
genre_grid.head()
A = genre_grid[0].value_counts().sort_index()
B = genre_grid[1].value_counts().sort_index()
C = genre_grid[2].value_counts().sort_index()
D = genre_grid[3].value_counts().sort_index()
E = genre_grid[4].value_counts().sort_index()
sumAB = A.add(B,fill_value = 0)
sumCD = C.add(D,fill_value = 0)
count_of_each_genre = (sumAB + sumCD).add(E, fill_value = 0)
count_of_each_genre
count_of_each_genre.plot.bar(title = "Movie Count Per Genre").set(xlabel="Genres", ylabel="Number of Movies")
