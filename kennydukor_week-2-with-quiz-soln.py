## all imports

from urllib.request import urlopen

from IPython.display import HTML

import numpy as np

#import urllib2

import bs4 #this is beautiful soup

import time

import operator

import socket

#import cPickle

import re # regular expressions



from pandas import Series

import pandas as pd

from pandas import DataFrame



import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set_context("talk")

sns.set_style("white")


# pass in column names for each CSV

user_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']



users = pd.read_csv(

    'http://files.grouplens.org/datasets/movielens/ml-100k/u.user', 

    sep='|', names=user_cols)



users.head()
ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings = pd.read_csv(

    'http://files.grouplens.org/datasets/movielens/ml-100k/u.data', 

    sep='\t', names=ratings_cols)



ratings.head()
# the movies file contains columns indicating the movie's genres

# let's only load the first five columns of the file with usecols

movie_cols = ['movie_id', 'title', 'release_date', 

            'video_release_date', 'imdb_url']



movies = pd.read_csv('http://files.grouplens.org/datasets/movielens/ml-100k/u.item', sep='|', 

                     names=movie_cols, usecols=range(5), encoding = "ISO-8859-1")

movies.head()
print(movies.dtypes)

print (movies.describe())

# *** Why only those two columns? ***
A = users.head()

B = users['occupation'].head()



columns_you_want = ['occupation', 'sex'] 

D = users[columns_you_want].head()



print (A, "\n========")

print (B, "\n========")

print (D)
oldUsers = users[users['age'] > 25]

oldUsers.head()
# users aged 40 AND male

# your code here

users[(users.age == 40) & (users.sex == "M")].head(3)
users[(users["age"] == 40) & (users["sex"] == "M")].head(3)
# but what if i want to see only age and sex

# your code here

columns_you_want_2 = users[['age', 'sex']]

print(columns_you_want_2.head())
#solution_cont

older = columns_you_want_2[(columns_you_want_2.age == 40) & (columns_you_want_2['sex'] == 'M')]

older.head()
## users who are female and programmers

# your code here



## show statistic summary or compute mean

# your code here

users['occupation'].unique()
females = users[(users.sex == "F") & (users.occupation == "programmer")].head(5)

females
#If we want only those column (sex, occupation)

columns_you_want = users[['sex', 'occupation']]

print(columns_you_want.head())
#solution_cont

occ = columns_you_want[(columns_you_want['sex'] == 'F') & (columns_you_want['occupation'] == 'programmer')]

occ.head()
# a smarter way

columns_you_want_better = females[['sex', 'occupation']]

columns_you_want_better
print (ratings.head())

print("=======")


## split data

#grouped_data = ratings.groupby('user_id')

grouped_data = ratings['movie_id'].groupby(ratings['user_id'])

#print(grouped_data.head(5))



## count and combine

ratings_per_user = grouped_data.count()



ratings_per_user.head(5)
#Other method

ratings.set_index(["user_id", "movie_id"]).count(level="user_id").head()
ratings.count()
## split data



# your code here

grouped_data_1 = ratings['rating'].groupby(ratings['movie_id'])

grouped_data_1.head(2)
## average and combine

# your code here\

average_ratings = grouped_data.mean()

print ("Average ratings:")

print (average_ratings.head())
# get the maximum rating

# your code here

maximum_rating = average_ratings.max()

maximum_rating
# get movie ids with that rating

# your code here

good_movie_ids = average_ratings[average_ratings == maximum_rating].index

good_movie_ids
print ("Good movie ids:")

print #your code here

print (good_movie_ids)

print ("===============\n=============")

print ("Best movie titles")

print # your code here

print (movies[movies.movie_id.isin(good_movie_ids)].title)
# get number of ratings per movie

# your code here

how_many_ratings = grouped_data.count()

print ("Number of ratings per movie")

print # your code here

print (how_many_ratings[average_ratings == maximum_rating])
average_ratings = grouped_data.apply(lambda f: f.mean())

average_ratings.head()
# get the average rating per user

# your code here

grouped_data = ratings['rating'].groupby(ratings.user_id)

average_ratings = grouped_data.mean()

average_ratings.head()
# list all occupations and if they are male or female dominant

# your code here
grouped_data = users['sex'].groupby(users['occupation'])

male_dominant_occupations = grouped_data.apply(lambda f: 

                                               sum(f == 'M') > sum(f == 'F'))

print (male_dominant_occupations)

print ('\n')
print ('number of male users: ')

print (sum(users['sex'] == 'M'))



print ('number of female users: ')

print (sum(users['sex'] == 'F'))


htmlString = """<!DOCTYPE html>

<html>

  <head>

    <title>This is a title</title>

  </head>

  <body>

    <h2> Test </h2>

    <p>Hello world!</p>

  </body>

</html>"""



htmlOutput = HTML(htmlString)

htmlOutput
import requests as req
url = 'http://www.crummy.com/software/BeautifulSoup'

source = req.get(url)

print (source.status_code) #To check that a request is successful, use r.raise_for_status() or check r.status_code is what you expect.
print(source.headers)
print(source.headers["Content-Type"])
print(source.text[:480])
params = {"query": "python download url content", "source":"chrome"}

source2 = req.get("http://www.google.com/search", params=params)

print(source2.status_code)
re.findall(r"Soup", source.text)
soup = re.search(r"Soup", source.text)

print(soup)
## get bs4 object

soup = bs4.BeautifulSoup(source.text)
## compare the two print statements

print (soup)

#print soup.prettify()
## show how to find all a tags

soup.findAll('a')



## ***Why does this not work? ***

#soup.findAll('Soup')
## get attribute value from an element:

## find tag: this only returns the first occurrence, not all tags in the string

first_tag = soup.find('a')

print(first_tag)

## get attribute `href`

print(first_tag.get('href'))



## get all links in the page

link_list = [l.get('href') for l in soup.findAll('a')]

link_list



# or

# link_list = []

# for l in soup.findAll('a'):

#     link_list.append(l.get('href'))

# link_list
# So, to find all the soup. We search within the tags



link_list = [l.get('Soup') for l in soup.findAll('html')]

link_list
# Get your own at https://github.com/settings/tokens/new

token = "" 

response = req.get("https://api.github.com/user", params={"access_token":token})



#print(response.status_code)

print(response.headers["Content-Type"])

print(response.json().keys())
response = req.get("https://api.github.com/user", auth=("kennydukor@gmail.com", "github_password"))

print(response.status_code)

print(response.headers["Content-Type"])

print(response.json())

print(response.json().keys())
print(response.content)
import json

print(json.loads(response.content))
data = {"a":[1,2,3,{"b":2.1}], 'c':4}

json.dumps(data)
#json.dumps(response)
a = {'a': 1, 'b':2}

s = json.dumps(a)

a2 = json.loads(s)



## a is a dictionary

print (a)

## vs s is a string containing a in JSON encoding

print (s)

## reading back the keys are now in unicode

print (a2)