# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# reading the reshaped Netflix dataset created as part of Assignment A5

df_netflix = pd.read_csv('/kaggle/input/netflix/reshaped_US_NetflixData.csv')



# Viewing the dataset

df_netflix
# This code retrieves the key from Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("omdbApi")



import urllib.error, urllib.parse, urllib.request, json



def safeGet(url):

    try:

        return urllib.request.urlopen(url)

    except urllib.error.URLError as e:

        if hasattr(e,"code"):

            print("The server couldn't fulfill the request.")

            print("Error code: ", e.code)

        elif hasattr(e,'reason'):

            print("We failed to reach a server")

            print("Reason: ", e.reason)

        return None



# defining a function to retrieve data from IMDB API   

def getMovieDetails(title): # passing title as a parameter

    key = secret_value_0

    url = "http://www.omdbapi.com/?t="+title+"&apikey="+key # assigning title and key in the API query URL

    url = url.replace (' ','%20') # replacing spaces with %20 for the API query to work

    return safeGet(url)
# creating arrays to store values from the API

omdb_title = []

imdbID = []

imdbVotes = []

imdbRating = []

metascore = []



# creating a dataframe to store extracted information from the API

df_omdb = pd.DataFrame()



# iterate through the Netflix dataset and assign 'Title' to the 'item' variable

for item in df_netflix['Title']:

    

    #loading the retuned JSON information to a variable data

    data = json.load(getMovieDetails(item))

   

    # storing the returned JSON information in the defined arrays

    omdb_title.append(data['Title'])

    imdbID.append(data['imdbID'])

    imdbVotes.append(data['imdbVotes'])

    imdbRating.append(data['imdbRating'])

    metascore.append(data['Metascore'])



# inserting the values to the omdb dataframe

df_omdb['Title'] = omdb_title

df_omdb['IMDB ID'] = imdbID

df_omdb['IMDB Votes'] = imdbVotes

df_omdb['IMDB Rating'] = imdbRating

df_omdb['Meta Critic Score'] = metascore



# viewing the data

df_omdb.head()
# Setting colummn Title as index

df_netflix = df_netflix.set_index('Title')



# Setting colummn Title as index

df_omdb = df_omdb.set_index('Title')



# merging the netflix and omdb dataframe using 'Title' column as index for the join

df_merged = pd.merge(df_netflix, df_omdb, on='Title')



#viewing the data

df_merged.head()
# writing the merged data in a new file

df_merged.to_csv('merged_US_NetflixOmdbData.csv')



# reading the merged dataset because API has a limit of 1000 calls

df_merged = pd.read_csv('/kaggle/input/merged-data/merged_US_NetflixOmdbData.csv')
import pandas_datareader.data as web

import matplotlib.pyplot as plt

import datetime as dt



# making the visulaizations bigger

plt.style.use('seaborn-poster')
# Viewing the data types

df_merged.dtypes
#Converting the data types to required format

df_merged['Netflix ID'] = df_merged['Netflix ID'].astype('str')

df_merged['Release Year'] = df_merged['Release Year'].astype('object')

df_merged['IMDB Votes'] = df_merged['IMDB Votes'].str.replace(',', '')

df_merged['IMDB Votes'] = df_merged['IMDB Votes'].astype(float)

df_merged['Duration'] = df_merged['Duration'].str.replace('min', '')

df_merged['Duration'] = df_merged['Duration'].str.replace('Seasons', '')

df_merged['Duration'] = df_merged['Duration'].astype(int)

df_merged['IMDB Rating'] = df_merged['IMDB Rating'].astype(float)



df_merged.dtypes
# viewing the dataset

df_merged.head()
# formatting the values

pd.options.display.float_format = '{:.2f}'.format



# Printing descriptive statistical values

df_merged.describe()
# finding the movie with highest IMDB rating

df_merged.loc[df_merged['IMDB Rating'].idxmax()]
# finding the movie with lowest IMDB rating

df_merged.loc[df_merged['IMDB Rating'].idxmin()]
# finding the movie with highest IMDB votes

df_merged.loc[df_merged['IMDB Votes'].idxmax()]
# finding the movie with lowest IMDB votes

df_merged.loc[df_merged['IMDB Votes'].idxmin()]
# finding the movie with highest Metascore

df_merged.loc[df_merged['Meta Critic Score'].idxmax()]
# finding the movie with lowest Metascore

df_merged.loc[df_merged['Meta Critic Score'].idxmin()]
# finding the longest duration movie

df_merged.loc[df_merged['Duration'].idxmax()]
# Choosing movies released in past 20 years

df_20Years = df_merged[df_merged['Release Year'] >= 2000]



# Counting the movies released in each year

df_count = df_20Years['Release Year'].value_counts()



#Defining display label for x-axis

plt.xlabel('Number')



#Defining display label for y-axis

plt.ylabel('Release Year')



# plotting a horizontal bar graph

df_count.plot(kind='barh', title = 'Number of Movies and TV shows released in past 20 years')
# counting the number of movies and tv shows on Netflix

df_count = df_merged['Type'].value_counts()



#Defining display label for x-axis

plt.xlabel('Type of content')



#Defining display label for y-axis

plt.ylabel('Number')



# plotting the vertical bar graph

df_count.plot(kind='bar', title = 'Number of Movies vs. TV shows on Netflix')
#Finding number of IMDB votes year over year

New_votes = np.array(df_merged['IMDB Votes'].groupby(df_merged['Release Year']).sum())



#Extracting years in array format from the dataframe

Years = np.sort(df_merged['Release Year'].unique())



#Plotting total IMDB votes each year in green color 

plt.plot(Years, New_votes, color='orange')



#Defining display label for x-axis

plt.xlabel('Year')



#Defining display label for y-axis

plt.ylabel('Number of IMDB Votes')



#Defining title for the plot

plt.title('IMDB votes YOY')



plt.show()
# scatter plot of IMDB Rating and Meta Critic Score

plt.plot(df_merged['IMDB Rating'], df_merged['Meta Critic Score'], 'go')



#Defining display label for x-axis

plt.xlabel('IMDB Rating - out of 10')



#Defining display label for y-axis

plt.ylabel('Meta Critic Score - out of 100')



#Defining title for the plot

plt.title('Correlation of IMDB Rating vs. Meta Critic Score')



plt.show()

# scatter plot of IMDB Rating and IMDB Votes

plt.plot(df_merged['IMDB Rating'], df_merged['IMDB Votes'], 'go')



#adjusting the Y axis range to make the graph readable

scale_factor = .001

xmin, xmax = plt.xlim()

ymin, ymax = plt.ylim()



plt.xlim(xmin, xmax)

plt.ylim(ymin * scale_factor, ymax * scale_factor)



#Defining display label for x-axis

plt.xlabel('IMDB Rating - out of 10')



#Defining display label for y-axis

plt.ylabel('IMDB Votes - in thousands')



#Defining title for the plot

plt.title('Correlation of IMDB Rating vs. IMDB Votes')



plt.show()
# finding top 10 movies with highest IMDB ratings

df_imdbtop = df_merged.nlargest(10, ['IMDB Rating'])



df_imdbtop.sort_values('IMDB Rating',inplace=True)

# plotting a horizontal bar graph

df_imdbtop.plot.barh(x='Title', y='IMDB Rating', rot=0, color = 'blue', alpha = 0.3)



#Defining title for the plot

plt.title ("Top 10 movies with highest IMDB ratings")



#Defining display label for x-axis

plt.xlabel("Title")



#Defining display label for y-axis

plt.ylabel("IMDB Rating")

# finding top 10 movies with highest IMDB votes

df_topvotes = df_merged.nlargest(10, ['IMDB Votes'])



df_topvotes.sort_values('IMDB Votes',inplace=True)



# plotting a horizontal bar graph

df_topvotes.plot.barh(x='Title', y='IMDB Votes', rot=0, color = 'seagreen', alpha = 0.4)



#Defining title for the plot

plt.title ("Top 10 movies with highest IMDB votes")



#Defining display label for x-axis

plt.xlabel("Title")



#Defining display label for y-axis

plt.ylabel("IMDB Votes")
# finding top 10 movies with highest Meta Critic Score

df_topmeta = df_merged.nlargest(10, ['Meta Critic Score'])



df_topmeta.sort_values('Meta Critic Score',inplace=True)



# plotting a horizontal bar graph

df_topmeta.plot.barh(x='Title', y='Meta Critic Score', rot=0, color = 'gold', alpha = 0.4)



#Defining title for the plot

plt.title ("Top 10 movies with highest Meta Critic Score")



#Defining display label for x-axis

plt.xlabel("Title")



#Defining display label for y-axis

plt.ylabel("Meta Critic Score")
# finding top 20 directors with highest IMDB ratings

df_topDir = df_merged.nlargest(20, ['IMDB Rating'])



#sorting data based on IMDB rating

df_topDir.sort_values('IMDB Rating',inplace=True)



# plotting a horizontal bar graph

df_topDir.plot.barh(x='Director', y='IMDB Rating', rot=0, color = 'seagreen', alpha = 0.4)



#Defining title for the plot

plt.title ("Top 20 directors with highest IMDB ratings")



#Defining display label for x-axis

plt.xlabel("IMDB Rating (out of 10)")



#Defining display label for y-axis

plt.ylabel("Director")
# creating a new dataframe

df_2019 = pd.DataFrame()



# iterate through the dataframe

for index, row in df_merged.iterrows():

        if row['Release Year'] == 2019: # checking if movie release in 2019

            if row['IMDB Rating'] > 8: # checking if IMDB rating > 8

                df_2019 = df_2019.append(row) # appending the row in dataframe



#sorting data based on IMDB rating

df_2019.sort_values('IMDB Rating',inplace=True)



# plotting a horizontal bar graph

df_2019.plot.barh(x='Title', y='IMDB Rating', rot=0, color = 'seagreen', alpha = 0.4)



#Defining title for the plot

plt.title ("Movies released in 2019 with IMDB rating greater than 8")



#Defining display label for x-axis

plt.xlabel("IMDB Rating (out of 10)")



#Defining display label for y-axis

plt.ylabel("Title")