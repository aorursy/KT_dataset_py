import pandas as pd

import numpy as np



import requests

from requests import get

from bs4 import BeautifulSoup



from time import sleep

from random import randint
# Creating the lists we want to write into

titles = []

years = []

time = []

imdb_ratings = []

metascores = []

votes = []

us_gross = []
# Getting English translated titles from the movies

headers = {'Accept-Language': 'en-US, en;q=0.5'}
pages = np.arange(1, 1001, 50)

pages
# Storing each of the urls of 50 movies 

for page in pages:

    # Getting the contents from the each url

    page = requests.get('https://www.imdb.com/search/title/?groups=top_1000&start=' + str(page) + '&ref_=adv_nxt', headers=headers)

    soup = BeautifulSoup(page.text, 'html.parser')

    

    # Aiming the part of the html we want to get the information from

    movie_div = soup.find_all('div', class_='lister-item mode-advanced')

    

    # Controling the loop’s rate by pausing the execution of the loop for a specified amount of time

    # Waiting time between requests for a number between 2-10 seconds

    sleep(randint(2,10))

    

    for container in movie_div:

        # Scraping the movie's name

        name = container.h3.a.text

        titles.append(name)

        

        # Scraping the movie's year

        year = container.h3.find('span', class_='lister-item-year').text

        years.append(year)

        

        # Scraping the movie's length

        runtime = container.find('span', class_='runtime').text if container.p.find('span', class_='runtime') else '-'

        time.append(runtime)

        

        # Scraping the rating

        imdb = float(container.strong.text)

        imdb_ratings.append(imdb)

        

        # Scraping the metascore

        m_score = container.find('span', class_='metascore').text if container.find('span', class_='metascore') else '-'

        metascores.append(m_score)

        

        # Scraping votes and gross earnings

        nv = container.find_all('span', attrs={'name':'nv'})

        vote = nv[0].text

        votes.append(vote)

        grosses = nv[1].text if len(nv) > 1 else '-'

        us_gross.append(grosses)
movies = pd.DataFrame({'movie':titles,

                       'year':years,

                       'time_minute':time,

                       'imdb_rating':imdb_ratings,

                       'metascore':metascores,

                       'vote':votes,

                       'gross_earning':us_gross})



movies.head()
movies.dtypes
# Cleaning 'year' column

movies['year'] = movies['year'].str.extract('(\d+)').astype(int)

movies.head(3)
# Cleaning 'time_minute' column

movies['time_minute'] = movies['time_minute'].str.extract('(\d+)').astype(int)

movies.head(3)
# Cleaning 'metascore' column

movies['metascore'] = movies['metascore'].str.extract('(\d+)')

# convert it to float and if there are dashes turn it into NaN

movies['metascore'] = pd.to_numeric(movies['metascore'], errors='coerce')
# Cleaning 'vote' column

movies['vote'] = movies['vote'].str.replace(',', '').astype(int)

movies.head(3)
# Cleaning 'gross_earning' column

# left strip $ and right strip M 

movies['gross_earning'] = movies['gross_earning'].map(lambda x: x.lstrip('$').rstrip('M'))

# convert it to float and if there are dashes turn it into NaN

movies['gross_earning'] = pd.to_numeric(movies['gross_earning'], errors='coerce')

movies.head(3)
movies.dtypes
movies
movies.to_csv('movies.csv')