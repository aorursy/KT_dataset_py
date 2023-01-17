# Filtering out the warnings



import warnings



warnings.filterwarnings('ignore')
# Importing the required libraries



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

# Read the csv file using 'read_csv'. Please write your dataset location here.



inp0=pd.read_csv('../input/imdb-movie-assignment/MovieAssignmentData (2).csv')

# Check the number of rows and columns in the dataframe



inp0.shape
# Check the column-wise info of the dataframe



inp0.info()
# Check the summary for the numeric columns 

inp0.describe()

# Divide the 'gross' and 'budget' columns by 1000000 to convert '$' to 'million $'

inp0['budget']=inp0['budget'] / 1000000

inp0['Gross']=inp0['Gross'] / 1000000

inp0
# Create the new column named 'profit' by subtracting the 'budget' column from the 'gross' column

inp0['profit']=inp0['Gross']-inp0['budget']

inp0
# Sort the dataframe with the 'profit' column as reference using the 'sort_values' function. Make sure to set the argument

#'ascending' to 'False'

inp0=inp0.sort_values(by='profit',ascending=False)

inp0.head()
# Get the top 10 profitable movies by using position based indexing. Specify the rows till 10 (0-9)



inp0.iloc[0:10]
#Plot profit vs budget



sns.jointplot('budget', 'profit', inp0)

plt.show()
#Find the movies with negative profit

inp0['Title'][inp0['profit']<0]
# Change the scale of MetaCritic

inp0.MetaCritic=inp0.MetaCritic/10
inp0.head()
# Find the average ratings

inp0['Avg_rating']=(inp0['IMDb_rating']+inp0['MetaCritic'])/2



inp0.head()
#Sort in descending order of average rating

inp0=inp0.sort_values(by=['Avg_rating'],ascending=False)
inp0.head()
# Find the movies with metacritic-rating < 0.5 and also with the average rating of >8

df=inp0[['Title','IMDb_rating','MetaCritic','Avg_rating']]

df=df.loc[(abs(df.IMDb_rating-df.MetaCritic)<0.5)]

df.loc[df.Avg_rating>=8]

# Write your code here



group = inp0.pivot_table(values = ['actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes'],

                           index = ['actor_1_name', 'actor_2_name', 'actor_3_name'])



group['Total likes'] = group['actor_1_facebook_likes'] + group['actor_2_facebook_likes'] + group['actor_3_facebook_likes']



group
group.sort_values(by=['Total likes'], inplace=True, ascending = False)



group.head(5)
# Your answer here (optional)



# Reset the index of the grouped dataframe so you can access the indices as columns easily

group.reset_index(inplace=True)



# Initialise the value of a variable 'j' to 0. This variable will be used to keep

# a track of the rows during the loop iteration



j = 0



# Run a loop through the length of the column 'Total likes'

for i in group['Total likes']:   

    temp = sorted([group.loc[j,'actor_1_facebook_likes'], group.loc[j,'actor_2_facebook_likes'], group.loc[j,'actor_3_facebook_likes']])

    if temp[0] >= temp[1]/2 and temp[0] >= temp[2]/2 and temp[1] >= temp[2]/2:

        print(sorted([group.loc[j, 'actor_1_name'], group.loc[j, 'actor_2_name'], group.loc[j, 'actor_3_name']]))

        break 

    j += 1 

# Runtime histogram/density plot



sns.distplot(inp0['Runtime'])
# Write your code here

PopularR=inp0[inp0.content_rating=='R'].sort_values(by='CVotesU18',ascending=False)[['Title','CVotesU18']].head(10)

PopularR
# Create the dataframe df_by_genre



df_by_genre=inp0[inp0.columns[11:-6]] 
# Create a column cnt and initialize it to 1

df_by_genre['cnt']=1

df_by_genre
# Group the movies by individual genres



df_by_g1 = df_by_genre.groupby('genre_1').sum()



df_by_g2 = df_by_genre.groupby('genre_2').sum()



df_by_g3 = df_by_genre.groupby('genre_3').sum()



# Add the grouped data frames and store it in a new data frame

df_all_genres = df_by_g1.add(df_by_g2, fill_value=0)

df_all_genres=df_all_genres.add(df_by_g3, fill_value=0)

df_all_genres

# Extract genres with atleast 10 occurences

genre_top10=df_all_genres[df_all_genres['cnt']>=10]

# Take the mean for every column by dividing with cnt 



for i in df_all_genres.columns[:-1]:

    df_all_genres[i]=df_all_genres[i]/df_all_genres['cnt']

df_all_genres

# Rounding off the columns of Votes to two decimals

for i in df_all_genres.columns:

    df_all_genres[i]=df_all_genres[i].round(2)

df_all_genres
# Converting CVotes to int type



for i in df_all_genres.columns:

    if i.startswith('CVotes'):

        df_all_genres[i]=df_all_genres[i].astype('int')

df_all_genres.info()
# Countplot for genres



sns.barplot(df_all_genres['cnt'],df_all_genres.index,data=df_all_genres)

# 1st set of heat maps for CVotes-related columns

fig, (ax1, ax2) = plt.subplots(1,2)

sns.heatmap(genre_top10[['CVotesU18M','CVotes1829M','CVotes3044M','CVotes45AM']],ax=ax1)

sns.heatmap(genre_top10[['CVotesU18F','CVotes1829F','CVotes3044F','CVotes45AF']],ax=ax2)

plt.show()
# 2nd set of heat maps for Votes-related columns





fig, (ax1, ax2) = plt.subplots(1,2)

sns.heatmap(genre_top10[genre_top10.columns[29:-1][genre_top10[genre_top10.columns[29:-1]].columns.str.contains('M')]],ax=ax1)

sns.heatmap(genre_top10[genre_top10.columns[29:-1][genre_top10[genre_top10.columns[29:-1]].columns.str.contains('F')]],ax=ax2)

plt.show()



# Creating IFUS column



inp0['IFUS']=inp0.Country.apply(lambda x: 'USA' if x == 'USA' else 'non-USA')
inp0['IFUS']
# Box plot - 1: CVotesUS(y) vs IFUS(x)



sns.boxplot(y='CVotesUS',x='IFUS',data=inp0)
# Box plot - 2: VotesUS(y) vs IFUS(x)



sns.boxplot(y='VotesUS',x='IFUS',data=inp0)

# Sorting by CVotes1000



genre_top10.sort_values(by='CVotes1000',ascending=False,inplace=True)

# Bar plot

sns.barplot(genre_top10['CVotes1000'],genre_top10.index,data=genre_top10)
