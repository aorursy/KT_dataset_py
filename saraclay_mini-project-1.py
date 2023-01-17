# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import datetime as dt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Create URL to JSON file (alternatively this can be a filepath)

url = '/kaggle/input/my-spotify-streaming-history/StreamingHistory0.json'



# Load the first sheet of the JSON file into a data frame

df = pd.read_json(url, orient='columns')



# View the first ten rows

df.head(10)
# It looks like I'd need to get the artist's URI first. THEN I can get the categories of the artist from the Spotify API.

# https://developer.spotify.com/documentation/web-api/reference/artists/get-artist/

# https://developer.spotify.com/console/get-artist/?id=0OdUWJ0sBjDrqHygGUXeCF



# I'm also going to use this handy Spotify module that I found

# Spotipy library - https://spotipy.readthedocs.io/en/2.9.0/
# Code help from - https://medium.com/@RareLoot/extracting-spotify-data-on-your-favourite-artist-via-python-d58bc92a4330

# Note that the following command needs to be run each time this project is opened

# For console: pip install spotipy --upgrade



# Importing the "spotipy" Python module

import spotipy

# This is added to access authorized Spotify data

from spotipy.oauth2 import SpotifyClientCredentials



# Client ID and secret are needed to get information

client_id = "38938e1ee8de47ceb9c3a0b368a70bbf"

# client_secret = {spotify secret id}

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

client_secret = user_secrets.get_secret("Spotify")



client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)

# Spotify object to access API

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) 

# Chosen artist

name = "Depeche Mode" 

# Search query

result = sp.search(name) 

# Code given to use as an example in Medium article

result['tracks']['items'][0]['artists']

# Search query

result = sp.search(name) 



# What happens if I print everything out? I commented this out after the fact.

# print(result)
# Search query

result = sp.search(name) 



# This is from the example

# result['tracks']['items'][0]['artists']

result['tracks']['items'][0]['artists'][0]['uri']
# I want to create a new column, so I'll create a variable with open brackets

# artistURI = []



# For each entry, I want to loop through and add the URI for each artist in each row



# row=0

# for artist in df['artistName']:

#     This will get the relevant info of the artist

#     result = sp.search(df['artistName'][row])

#     artistURI.append(result['tracks']['items'][row]['artists'][0]['uri'])

#     row = row + 1



# # Create a column from the list

# df['artistURI'] = artistURI 
# Get the unique names of the artists

unique_names = df['artistName'].head(20).unique()

print (unique_names)
artistURI = []



# For each entry, I want to loop through and add the URI for each artist in each row

row=0



for artist in unique_names:

    full_result = sp.search(unique_names[row])

    artistURI.append(full_result['tracks']['items'][row]['artists'][0]['uri'])

    row = row + 1

    

print (artistURI)
# This is the first line of code I tried, but this gave me this info for every artist

# df.groupby('artistName').size()



# This limits the artists to the top 20 but I can't see the count

# n = 20

# top_artists = df['artistName'].value_counts()[:n].index.tolist()

# top_artists



# Now I'm taking the top-played artists in descending order and only looking at however many I want to!

top_artists = df['artistName'].value_counts(ascending=False)

print ("These were the artists I listened to the most in 2019: ")

top_artists.head(20)
plt.style.available
plt.style.use("seaborn-darkgrid")

top_artists.head(20).plot(kind = 'bar', figsize=(30,5), rot = 0)
df.head(1)
df['endTime'] = pd.to_datetime(df['endTime'])

df['endTime'].head(5)
df['day_of_week'] = df['endTime'].dt.day_name()

df['day_of_week']
week_listening = df['day_of_week'].value_counts(ascending=False)

week_listening.head(7)
plt.style.use("dark_background")

week_listening.head(7).plot(kind = 'bar', figsize=(15,5), rot = 0)
# This converts endTime to datetime format

df['hour'] = pd.to_datetime(df['endTime'])

df['hour'].head(5)