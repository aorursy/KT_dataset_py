# Import python packages i plan to use.

import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
% matplotlib inline
# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.

df = pd.read_csv('../input/tmbd-movies-dataset-udacity/tmdb-movies.csv')
df.head(3)
# To check the size of the dataset
df.shape
df.info()
df.describe()
df.isnull().sum()
#Number of duplicate rows
df.duplicated().sum()
# Make sure of duplicate row is exactly the same 
df[df.duplicated(keep=False)]
df.nunique()
df.isnull().any(axis=1).sum()
col_with_zero = ['runtime','budget_adj','revenue_adj']
for i in col_with_zero:
    zero_count = (df[i] == 0).sum()
    print('`{}` have {} zero values'.format(i,zero_count))
# # drop columns from dataset.
df.drop(['id','imdb_id','homepage','tagline','keywords','overview'
         ,'budget','revenue','vote_count'],axis = 1,inplace = True)

# showe the resualt.
df.head(3)
# drop duplicate rows
df.drop_duplicates(inplace=True)
# check number of duplicates -it should be 0
df.duplicated().sum()
# creeate a list columns with zero values.
col_with_zero = ['runtime','budget_adj','revenue_adj']

# replace zero values with NaN for columns in the list.
df[col_with_zero] = df[col_with_zero].replace(0,np.NAN)

# confirme the changes
df.describe()
# fill NaN values with mean
df['runtime'].fillna(df['runtime'].mean(),inplace = True)
df['budget_adj'].fillna(df['budget_adj'].mean(),inplace = True)
df['revenue_adj'].fillna(df['revenue_adj'].mean(),inplace = True)
# adding new column profit calculated using revenue minus budget 
df['profit'] = df['revenue_adj'] - df['budget_adj']
# confirm changes
df.head(1)
# convert release_data to datetime formate
df['release_date']=pd.to_datetime(df['release_date'])
# confirm changes
df.dtypes
# save cleaned data for next steps 
df.to_csv('tmdb_cleaned_data.csv', index = False)
pd.plotting.scatter_matrix(df,figsize=(15,15));
df['profit'].corr(df['popularity'])
df['profit'].corr(df['runtime'])
df['profit'].corr(df['vote_average'])
df['profit'].corr(df['budget_adj'])
# create new dataframe by filter to movies that made profit of more than 100Million dollars 
high_profit_movie = df.query('profit >= 100000000')

high_profit_movie.head(3)
high_profit_movie.describe()
# detailes of highest profit movie
highest = high_profit_movie['profit'].idxmax()
highest_details = pd.DataFrame(high_profit_movie.loc[highest])
highest_details
# the average popularity of the movies
high_profit_movie['popularity'].mean()
# create histogram to see the distribution of the popularity 
plt.figure(figsize=(10,5), dpi = 100)
sns.set_style('darkgrid')
# x-axis 
plt.xlabel('Movie popularity(Million)', fontsize = 15)
# y-axis 
plt.ylabel('No. of Movies', fontsize=15)
# distribution title
plt.title('Movie popularity Distribution', fontsize=15)

# Plot the histogram
plt.hist(high_profit_movie['popularity'], rwidth = 0.9, bins =35)
# Displays the plot
plt.show()
# the average budget of the movies
high_profit_movie['budget_adj'].mean()
# the average revenue of the movies
high_profit_movie['revenue_adj'].mean()
# release year have highest profit
highest_profit_year = high_profit_movie.groupby('release_year')['profit'].sum()
highest_profit_year.idxmax()
# Figure size
plt.figure(figsize=(12,6), dpi = 130)
sns.set_style('darkgrid')
# x-axis
plt.xlabel('Year', fontsize = 12)
# y-axis
plt.ylabel('Profit', fontsize = 12)
# Title
plt.title('Higtest Profit Year')

# Plot line Chart
plt.plot(highest_profit_year)

# Display the line Chart
plt.show()
# create a new column month by extracting the month from the release date
high_profit_movie['month'] = high_profit_movie['release_date'].apply(lambda x: x.month)
# total of profits group by month 
highest_profit_month = high_profit_movie.groupby('month')['profit'].sum()
# count of high profit movies group by month
high_profit_movie_month = high_profit_movie.groupby('month')['profit'].count()

highest_profit_month
high_profit_movie_month
# get the month with the highest movies profit
highest_profit_month.idxmax()
# get the month with largest count of high profit movies
high_profit_movie_month.idxmax()
# Figure size
plt.figure(figsize=(15,8))
sns.set_style('darkgrid')

month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.bar([1,2,3,4,5,6,7,8,9,10,11,12], highest_profit_month, tick_label = month_name)
# Title
plt.title('Highest Profit Month')
# y-axis
plt.ylabel('Profit')
# x-axis
plt.xlabel('Month');
# Figure size
plt.figure(figsize=(15,8))
sns.set_style('darkgrid')

month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.bar([1,2,3,4,5,6,7,8,9,10,11,12], high_profit_movie_month, tick_label = month_name)
# Title
plt.title('Number of High Profit Movies Ber Months')
# y-axis
plt.ylabel('Number of Movies')
# x-axis
plt.xlabel('Months');
def extract_high_proft_data(column):

    data = high_profit_movie[column].str.cat(sep = '|')
    
    # create pandas series and store the values separately
    data = pd.Series(data.split('|'))
    
    # display value count in descending order
    count = data.value_counts(ascending = False)
    
    return count
# get top 10 casts
cast = extract_high_proft_data('cast')
cast.head(15)
# get top 10 directors
director = extract_high_proft_data('director')
director.head(10)
# Get top 10 production companies
production_companies = extract_high_proft_data('production_companies')
production_companies.head(10)
# Get top 10 genres
director = extract_high_proft_data('genres')
director.head(10)
# create new dataframe by filter to movies that made vote of more than or equal to 7.0   
high_vote_movie = df.query('vote_average >= 7.0')

high_vote_movie.head(3)
high_vote_movie.describe()
highest = high_vote_movie['vote_average'].idxmax()
highest_details = pd.DataFrame(high_vote_movie.loc[highest])
highest_details
# the average popularity of the movies
high_vote_movie['popularity'].mean()
# create histogram to see the distribution of the popularity 
plt.figure(figsize=(10,5), dpi = 100)
sns.set_style('darkgrid')
# x-axis 
plt.xlabel('Movie popularity', fontsize = 15)
# y-axis 
plt.ylabel('No. of Movies', fontsize=15)
# distribution title
plt.title('Movie popularity Distribution', fontsize=15)

# Plot the histogram
plt.hist(high_vote_movie['popularity'], rwidth = 0.9, bins =35)
# Displays the plot
plt.show()
# the average budget of the movies
high_vote_movie['budget_adj'].mean()
# the average revenue of the movies
high_vote_movie['revenue_adj'].mean()
def extract_high_vote_data(column):

    data = high_vote_movie[column].str.cat(sep = '|')
    
    # create pandas series and store the values separately
    data = pd.Series(data.split('|'))
    
    # display value count in descending order
    count = data.value_counts(ascending = False)
    
    return count
# get top 10 casts
vote_cast = extract_high_vote_data('cast')
vote_cast.head(15)
# get top 10 directors
vote_director = extract_high_vote_data('director')
vote_director.head(10)
# Get top 10 production companies
vote_production_companies = extract_high_vote_data('production_companies')
vote_production_companies.head(10)
# Get top 10 genres
vote_genres = extract_high_vote_data('genres')
vote_genres.head(10)
