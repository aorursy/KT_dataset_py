# Use this cell to set up import statements for all of the packages that you plan to use.
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Remember to include a 'magic word' so that your visualizations are plotted
#   inline with the notebook. See this page for more:
#   http://ipython.readthedocs.io/en/stable/interactive/magics.html
# Load data
moviedata = pd.read_csv('../input/tmdb-movies.csv')
# Get general info
moviedata.info()
# Get an overview
moviedata.head()
# Drop duplicates
moviedata.drop_duplicates(inplace=True)
# Check if done (-1 entry)
moviedata.info()
# Drop rows containing missing values in genres
moviedata.dropna(subset=['genres'], inplace=True)  
moviedata.info()
# Create variable profit
moviedata ['profit'] = moviedata['revenue'] - moviedata['budget']
# Only keep columns that are needed for further analysis using movie title as index
md = moviedata[['popularity','budget','revenue', 'original_title','runtime', 'genres','vote_count','vote_average','profit','release_year']]
# md.set_index('original_title', inplace=True)
# Check result
md.head()
# Split genres and create a new entry for each of the genre a movie falls into
s = md['genres'].str.split('|').apply(Series, 1).stack()
s.index = s.index.droplevel(-1)
s.name = 'genres'
del md['genres']
md_split_genres = md.join(s)

# Check result
md_split_genres.head()
# Check entries (should be a lot more rows since the most movies have more than one genre)
md_split_genres.shape
# Look at histograms to get idea of how variables are distrubuted (overall)
md.hist(color='DarkBlue',figsize= (10,10));
# Group data by genre and get mean for each genre and each variable, divide by 1 mio for clarity and better visibility
md_genre_mean = md_split_genres.groupby(['genres']).mean()
md_genre_mean ['profit_million'] = md_genre_mean['profit']/1000000
del md_genre_mean['profit']
md_genre_mean['revenue_million'] = md_genre_mean['revenue']/1000000
del md_genre_mean['revenue']
md_genre_mean['budget_million'] =md_genre_mean['budget']/1000000
del md_genre_mean['budget']
# Get distribution of mean of variables grouped by genre
md_genre_mean.hist(color='DarkBlue',figsize= (10,10));
# Overall Descriptive statistics
md.describe()
# Get movies with highest budget, profit, popularity
md.nlargest(3, 'budget')
md.nlargest(3, 'profit')
md.nlargest(3, 'popularity')
# Get movies made per year, create new data frame
md_year = pd.DataFrame(md_split_genres.groupby('release_year').original_title.nunique())
md_year.head()
# Get max of movies made per year
md_year.nlargest(5,'original_title')
# Plot data, line chart for showing development over the years
md_year.plot.line(title = 'Movies made per year',color='DarkBlue',figsize=(10, 8));
# Get mean of variables grouped by year (new data frame) in order to see what changed
md_year_mean = md_split_genres.groupby('release_year').mean()
# Check results
md_year_mean.head()
# plot the development of revenue, profit and budget of movies over the years
md_year_mean[['revenue','profit','budget']].plot(title = 'TBD',color=('DarkBlue','c','crimson'),linestyle=('-'),figsize=(10, 8));
md_year_mean[['vote_average', 'vote_count']].plot(title = 'TBD',color=('DarkBlue','c'),figsize=(10, 8),secondary_y=['vote_average']);
# Lets turn to genres, reminder of what the split looked like
md_split_genres.head()
# How many different genres do we have?
md_split_genres['genres'].unique()
len(md_split_genres['genres'].unique())
# Group movies by genre using title as unique identifier and display all genres.
md_genre = (pd.DataFrame(md_split_genres.groupby('genres').original_title.nunique())).sort_values('original_title', ascending=True)
md_genre.head(20)
md_genre['original_title'].plot.pie(title= 'Movies per Genre in %', figsize=(10,10), autopct='%1.1f%%',fontsize=15);
# Display in bar chart
md_genre['original_title'].plot.barh(title = 'Movies per Genre',color='DarkBlue', figsize=(10, 9));

# Check results
md_genre_mean.head()
# Sort data in acending order 
md_genre_mean.sort_values('budget_million', ascending=True, inplace = True )
# Create bar chart with revenue and budget
md_genre_mean[['revenue_million', 'budget_million']].plot.barh(stacked=False, title = 'Budget and Revenue by Genre (US$ million)',color=('DarkBlue','c'), figsize=(15, 10));
md_genre_mean.sort_values('profit_million', ascending=True, inplace = True )
md_genre_mean['profit_million'].plot.barh(stacked=False, title = 'Profit by Genre (US$ million)',color='DarkBlue', figsize=(10, 9));
md_genre_mean.sort_values('vote_average', ascending=True, inplace = True)
md_genre_mean[['vote_average']].plot.barh(stacked=True, title = 'Voting Avg by Genre',color='DarkBlue', figsize=(10, 9));
md_genre_mean.sort_values('popularity', ascending=True, inplace = True)
md_genre_mean[['popularity']].plot.barh(stacked=True, title = 'Genres by Avg Popularity',color='DarkBlue', figsize=(10, 9));

md_genre_mean.sort_values('vote_count', ascending=True, inplace = True)
md_genre_mean[['vote_count']].plot.barh(stacked=True, title = 'Genres by Avg Vote Count',color='DarkBlue',figsize=(10, 9));
md_8 = md_split_genres[md_split_genres['vote_average']>=8]
md_8 = (pd.DataFrame(md_split_genres.groupby('genres').original_title.nunique())).sort_values('original_title', ascending=True )
md_8[['original_title']].plot.barh(stacked=True, title = 'Genres with >= 8 ratings', figsize=(10, 9),color='DarkBlue');

# Reminder of how the data frame looked like, when we splitted for genres
md_split_genres.head()
# Create data frame grouped by genres AND release year, get means of variables of interest
md_year_genre_mean = pd.DataFrame(md_split_genres.groupby(['release_year','genres'])['revenue', 'budget','profit','vote_average','vote_count','popularity'].mean())
md_year_genre_mean.head()
# Create data frame for average profit per genre per year
md_year_genre_profit = pd.DataFrame(md_split_genres.groupby(['release_year','genres'])['profit'].mean())
md_year_genre_profit.head()
# pivot data to get the shape that is necessary for a heatmap that displays genres, years and avg. profit per genre per year
md_heat_profit_pivot = pd.pivot_table(md_year_genre_profit, values='profit', index=['genres'], columns=['release_year'])
md_heat_profit_pivot.head()
# display heatmap
sns.set(rc={'figure.figsize':(15,10)})
sns.heatmap(md_heat_profit_pivot, linewidths=.5, cmap='YlGnBu');
md_year_genre_revenue = pd.DataFrame(md_split_genres.groupby(['release_year','genres'])['revenue'].mean())
md_heat_revenue_pivot = pd.pivot_table(md_year_genre_revenue, values='revenue', index=['genres'], columns=['release_year'])
sns.set(rc={'figure.figsize':(15,10)})
sns.heatmap(md_heat_revenue_pivot, linewidths=.5, cmap='YlGnBu');
md_year_genre_budget = pd.DataFrame(md_split_genres.groupby(['release_year','genres'])['budget'].mean())
md_heat_budget_pivot = pd.pivot_table(md_year_genre_budget, values='budget', index=['genres'], columns=['release_year'])
sns.set(rc={'figure.figsize':(15,10)})
sns.heatmap(md_heat_budget_pivot, linewidths=.5, cmap='YlGnBu');
md_year_genre_vote_avg = pd.DataFrame(md_split_genres.groupby(['release_year','genres'])['vote_average'].mean())
md_heat_vote_avg_pivot = pd.pivot_table(md_year_genre_vote_avg, values='vote_average', index=['genres'], columns=['release_year'])
sns.set(rc={'figure.figsize':(15,10)})
sns.heatmap(md_heat_vote_avg_pivot, linewidths=.5, cmap='YlGnBu');
md_year_genre_vote_count = pd.DataFrame(md_split_genres.groupby(['release_year','genres'])['vote_count'].mean())
md_heat_vote_count_pivot = pd.pivot_table(md_year_genre_vote_count, values='vote_count', index=['genres'], columns=['release_year'])
sns.set(rc={'figure.figsize':(15,10)})
sns.heatmap(md_heat_vote_count_pivot, linewidths=.5, cmap='YlGnBu');
md_year_genre_pop = pd.DataFrame(md_split_genres.groupby(['release_year','genres'])['popularity'].mean())
md_heat_pop_pivot = pd.pivot_table(md_year_genre_pop, values='popularity', index=['genres'], columns=['release_year'])
sns.set(rc={'figure.figsize':(15,10)})
sns.heatmap(md_heat_pop_pivot, linewidths=.5, cmap='YlGnBu');
md.corr(method='pearson')
md.plot.scatter(x='vote_average', y='profit',title='Profit vs Vote Avg',color='DarkBlue',figsize=(6,5));
md.plot.scatter(x='vote_average', y='revenue',title='Revenue vs Vote Avg',color='DarkBlue',figsize=(6,5));
md.plot.scatter(x='vote_count', y='profit',title='Profit vs Vote Count', color='DarkBlue', figsize=(6,5));
md.plot.scatter(x='vote_count', y='revenue',title='Revenue vs Vote Count', color='DarkBlue', figsize=(6,5));
md.plot.scatter(x='popularity', y='profit',title='Profit vs Popularity', color='DarkBlue', figsize=(6,5));
md.plot.scatter(x='popularity', y='revenue',title='Revenue vs Popularity', color='DarkBlue', figsize=(6,5));
md.plot.scatter(x='revenue', y='budget',title='Budget vs Revenue', color='DarkBlue', figsize=(6,5));
md.plot.scatter(x='profit', y='budget',title='Budget vs Profit', color='DarkBlue', figsize=(6,5));
md.plot.scatter(x='vote_average', y='budget',title='Budget vs Vote Avg', color='DarkBlue', figsize=(6,5));
md.plot.scatter(x='popularity', y='budget',title='Budget vs Popularity', color='DarkBlue', figsize=(6,5));
