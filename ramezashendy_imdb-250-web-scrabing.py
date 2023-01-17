# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import requests

r = requests.get('https://www.imdb.com/chart/top?ref_=nv_mv_250')

from bs4 import BeautifulSoup

soup = BeautifulSoup(r.text, 'html.parser')
results = soup.find_all('td', attrs={'class':'titleColumn'})
records = []

for result in results:

    date = result.find('a')['title']

    #date = result.find('a')['href']

    records.append((date))
[x.strip() for x in records[0].split(',')]
from bs4 import BeautifulSoup

import requests

import re



# Download IMDB's Top 250 data

url = 'http://www.imdb.com/chart/top'

response = requests.get(url)

soup = BeautifulSoup(response.text, 'lxml')



movies = soup.select('td.titleColumn')

links = [a.attrs.get('href') for a in soup.select('td.titleColumn a')]

crew = [a.attrs.get('title') for a in soup.select('td.titleColumn a')]

ratings = [b.attrs.get('data-value') for b in soup.select('td.posterColumn span[name=ir]')]

votes = [b.attrs.get('data-value') for b in soup.select('td.ratingColumn strong')]



imdb = []



# Store each item into dictionary (data), then put those into a list (imdb)

for index in range(0, len(movies)):

    # Seperate movie into: 'place', 'title', 'year'

    movie_string = movies[index].get_text()

    movie = (' '.join(movie_string.split()).replace('.', ''))

    movie_title = movie[len(str(index))+1:-7]

    year = re.search('\((.*?)\)', movie_string).group(1)

    place = movie[:len(str(index))-(len(movie))]

    data = {"movie_title": movie_title,

            "year": year,

            "place": place,

            "star_cast": crew[index],

            "rating": ratings[index],

            "vote": votes[index],

            "link": links[index]}

    imdb.append(data)



for item in imdb:

    print(item['place'], '-', item['movie_title'], '('+item['year']+') -', 'Starring:', item['star_cast'])
item
pd.DataFrame(imdb)