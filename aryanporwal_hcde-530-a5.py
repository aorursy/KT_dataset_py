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
# This code retrieves the key from Kaggle Secret Keys file

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("omdbApi")



import urllib.error, urllib.parse, urllib.request, json



def safeGet(url):

    try:

        return urllib.request.urlopen(url)

    except urllib2.error.URLError as e:

        if hasattr(e,"code"):

            print("The server couldn't fulfill the request.")

            print("Error code: ", e.code)

        elif hasattr(e,'reason'):

            print("We failed to reach a server")

            print("Reason: ", e.reason)

        return None





#testing the connection with the OMDBAPI    

def getMovieDetails(imdbID="tt3896198"): # passing IMDB ID as parameter

    key = secret_value_0

    url = "http://www.omdbapi.com/?i="+imdbID+"&apikey="+key

    print(url)

    return safeGet(url)



#loading the retuned JSON information to a variable data

data = json.load(getMovieDetails()) 



# printing the JSON data

print(data) 
# Reading the Netflix dataset from Kaggle

df = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')

# Viewing the dataset

df
# Viewing the dimensionality of the dataframe

df.shape
# Keeping movies created in United States only in the dataframe

df = df[df['country'] == 'United States']

# Viewing the dataset

df
# Viewing the dimensionality of the dataframe

df.shape
# Removein country, date_added, rating, listed_in and description columns from the dataset

df = df.drop(['country', 'date_added', 'rating', 'listed_in', 'description'], axis=1)

# Viewing the dimensionality of the dataframe

df.shape
# Removing rows with null value

df = df.dropna(axis=0, how='any')



# Viewing the dimensionality of the dataframe

df.shape
# Viewing the dataset

df
# Renaming column titles to new values

df = df.rename(columns = {"show_id" : "Netflix ID", "type": "Type", 'title' : 'Title', 'director' : 'Director', 'cast' : 'Cast', 'release_year' : 'Release Year', 'duration' : 'Duration', 'description' : 'Description'})



# Setting colummn Netflix ID as index

df = df.set_index('Netflix ID')



# Sort by Release Year

df = df.sort_values(by = ['Release Year'])



# Viewing the dataset

df
# writing the changed data in a new file

df.to_csv('reshaped_US_NetflixData.csv')
import pandas_datareader.data as web

import matplotlib.pyplot as plt

import datetime as dt
# to make the visulaization bigger

plt.style.use('seaborn-poster')
# Choosing movies release in past 20 years

df_20Years = df[df['Release Year'] >= 2000]



# Counting the movies release in each year

df_count = df_20Years['Release Year'].value_counts()



# plotting a horizontal bar graph

df_count.plot(kind='barh', title = 'Number of movies and TV shows released in past 20 years')
# counting the type of movies and tv shows

df_count = df['Type'].value_counts()



# plotting the vertical bar graph

df_count.plot(kind='bar', title = 'Number of movies and TV shows on Netflix')