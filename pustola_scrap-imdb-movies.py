# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import requests

import re

import pandas as pd

from bs4 import BeautifulSoup

from IPython.display import clear_output

from dateutil import parser
def scrap_movies(page, progress_counter, release_country = 'USA'):

    data = {}   

       

    # 1. Get request model of imdb movies and convert it to str with .txt

    r_movies = requests.get(f'https://www.imdb.com/search/title/?title_type=feature,documentary&release_date=1980-01-01,&num_votes=1000,&sort=num_votes,desc&start={progress_counter}').text

    # 2. Use soup to parse html object

    s_movies = BeautifulSoup(r_movies, 'html.parser')



    # Looking for info on main list

    # Finding div with all movie info

    for index, div in enumerate(s_movies.find_all('div', class_='lister-item-content')):



        # Progress counter

        print('Page', page)

        print('Movies left:', 50-len(data)) 

        info = ({'title_id': None, 

                 'year': None, 

                 'release_date': None, 

                 'rating':None, 

                 'votes_number': None, 

                 'certificate':None, 

                 'runtime':None, 

                 'genres':None, 

                 'director':None, 

                 'cast': None, 

                 'original_title': None, 

                 'plot': None,

                 'countries': None})



        ###### Scrapping title ID and title: ######

        for h3 in div.find_all('h3'):

            #titles.append(h3.a.text)

            title_id = (h3.a.get('href'))

            info['title_id'] = title_id

            movie_url = 'https://www.imdb.com'+title_id

            print(f'Title: {h3.a.text}')



            ###### Scrapping year ######

            for span in h3.find_all('span', class_='lister-item-year'):

                year_str = span.text

                year = re.search('\d\d\d\d',year_str).group()

                info['year'] = int(year)



        ###### Scrapping certificate, runtime, genres ######

        for p in div.find_all('p'):



            ###### Certificates                

            for cert in p.find_all('span', class_='certificate'):

                info['certificate'] = cert.text



            ###### Runtime

            for runtime in p.find_all('span', class_='runtime'):

                info['runtime'] = int(re.search(r'\d*', runtime.text).group())



            ###### Genres

            for genre in p.find_all('span', class_='genre'):

                split = re.split(',', genre.text)

                sub = [re.sub('(\n)|( *)', '', x) for x in split]

                info['genres'] = sub



        ###### Scrapping ratings ######

        for rating in div.find_all('div', class_='ratings-bar'):

            info['rating'] = rating.strong.text



        ###### Scrapping vote numbers ######

        for p in div.find_all('p'):

            container = []

            for votes in p.find_all('span', attrs={"name": "nv"}):

                container.append(votes.text)

        # Getting every 2nd element from the list because name:'nv' contains also gross $$$

        info['votes_number'] = container[0]

        

        try:

            ###### Scrapping director ######

            print('Scrapping director')

            r_director = requests.get(movie_url).text

            s_director = BeautifulSoup(r_director, 'html.parser')



            div = s_director.find('div', class_='credit_summary_item')

            link = div.find_all('a')[0]

            info['director'] = link.text

        except Exception as ex:

            info['director'] = f'Error type: {type(ex).__name__}, Args: {ex.args}'



        try:

            ###### Scrapping release date (pass country to main func) ######

            print('Scrappint release date')

            release_url = movie_url+'releaseinfo'

            r_release = requests.get(release_url).text

            s_release = BeautifulSoup(r_release, 'html.parser')



            for tr in s_release.find_all('tr', class_='ipl-zebra-list__item release-date-item'):

                if (tr.find_all('td', class_='release-date-item__country-name')[0].text == f'{release_country}\n') & (not tr.find_all('td', class_='release-date-item__attributes')): #Second part makes sure that it wasn't a premiere on a festival

                    date_str = tr.find_all('td', class_='release-date-item__date')[0].text

                    release_date = parser.parse(date_str)

                    info['release_date'] = release_date

        except Exception as ex:

            info['release_date'] = f'Error type: {type(ex).__name__}, Args: {ex.args}'

        

        try:

            ###### Scrapping countries ######

            print('Scrapping countries')

            r_country = requests.get(movie_url).text

            s_country = BeautifulSoup(r_country, 'html.parser')



            countries = []

            for div in s_country.find_all('div', id='titleDetails'):

                for a in div.find_all('div', class_="txt-block")[1].find_all('a'):

                    countries.append(a.text)

            info['countries'] = countries

        except Exception as ex:

            info['countries'] = f'Error type: {type(ex).__name__}, Args: {ex.args}'

        

        try:

            ###### Scrapping cast ######

            print('Scrapping cast')

            # Getting cast url

            cast_url = movie_url+'fullcredits'

            r_cast = requests.get(cast_url).text

            s_cast = BeautifulSoup(r_cast, 'html.parser')

            # Scrapping cast from the web

            cast_list = s_cast.find_all('table', class_='cast_list')[0]

            tr = cast_list.find_all('tr')

            cast = []

            for i in tr:

                for td in i.find_all('td'):

                    if td.a:

                        if 'name' in td.a.get('href'):

                            cast.append(td.a.text)

            # Deleting every 2nd item from  the list since it's a NaN

            del cast[::2]

            # Subtructing whitespace from the begining and break point from the end of the actors name

            cast = [re.sub(r'(^ )|($\n)', '', x) for x in cast]

            info['cast'] = cast

        except Exception as ex:

            info['cast'] = f'Error type: {type(ex).__name__}, Args: {ex.args}'

        



        ###### Scrapping plot ######

        print('Scrapping plot')

        # 2. Link that leads to synopsis

        plot_url = movie_url+'plotsummary?ref_=tt_stry_pl#synopsis'

        # 3. Create working synopsis link and get request model

        r_synopsis = requests.get(plot_url).text

        # 4. Use soup to parse synopsis html object

        s_synopsis = BeautifulSoup(r_synopsis, 'html.parser')

        # 5. Extract synopsis text

        synopsis = s_synopsis.find('ul', id='plot-synopsis-content').li.text

        info['plot'] = synopsis



        ###### Scrapping original title. ######

        print('Scrapping original title')

        #If basic title = original it creates error. Get basic title.

        try:

            r_title = requests.get(movie_url).text

            s_title = BeautifulSoup(r_title, 'html.parser')

            original_title = s_title.find_all('div', class_='originalTitle')[0].text

            original_title = re.sub(' \(original title\)', '', original_title)

        except:

            original_title = s_title.find_all('div', class_='title_wrapper')

            original_title = re.search(r'[^\xa0]*',original_title[0].h1.text).group()

        info['original_title'] = original_title





        data[h3.a.text] = info

        clear_output()

    return data
path = r'YOUR_PATH_HERE'

# Scrapping movies in baches of 50 per imdb page i.e. (1,50,50) or (51,100,50)

for i in range(1, 9999, 50):

    page = (i, i+49)

    data = scrap_movies(page, i)

    

    movies = pd.DataFrame.from_dict(data.values())

    movies['title'] = data.keys()

    cols = ['title', 'original_title', 'year', 'release_date', 'rating', 'votes_number', 'runtime', 'certificate', 'countries', 'genres', 'plot', 'director', 'cast', 'title_id']

    movies = movies[cols]

    

    

    # Saving to csv

    print(f'Saving to movies_{page[0]}_{page[1]}.csv')

    movies.to_csv(f'{path}movies_{page[0]}_{page[1]}.csv', index=False)
# Reading .csv files in a loop and concatenating them to movie data frame

df_list = []

for i in range(1, 9900, 50):

    page = (i, i+49)

    df_list.append(pd.read_csv(f'{path}movies_{page[0]}_{page[1]}.csv'))

imdb_movies = pd.concat(df_list, ignore_index=True)

imdb_movies.to_csv('imdb_movies.csv', index=False)
# Checking where data wasn't available and 'Error' was put in instead

imdb_movies.applymap(lambda x: True if 'Error' in str(x) else False).sum()
# Getting rid of 'Error' string

imdb_movies['cast'] = imdb_movies['cast'].apply(lambda x: '["no_actors"]' if 'Error' in x else x)

imdb_movies['countries'] = imdb_movies['countries'].apply(lambda x: '["no_countries"]' if 'Error' in x else x)

imdb_movies['release_date'] = imdb_movies['release_date'].apply(lambda x: None if 'Error' in str(x) else x)
# Saving final data

imdb_movies.to_csv('imdb_movies.csv', index=False)