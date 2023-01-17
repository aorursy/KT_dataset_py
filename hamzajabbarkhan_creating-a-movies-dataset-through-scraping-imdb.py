!pip install pyforest
import pyforest
import requests
!pip install bs4
from bs4 import BeautifulSoup
response = requests.get('https://www.imdb.com/search/title/?release_date=2017-01-01,2017-12-31&sort=num_votes,desc&ref_=adv_prv')
response.headers
response.text[0:800]
content = response.text
parser = BeautifulSoup(content, 'html.parser')
test = parser.find('div', class_ = 'lister-item-content')

test
test_name = test.h3.a.text

test_name
test_ratings_imdb = test.find('div', class_ = 'inline-block ratings-imdb-rating')

test_ratings_imdb.strong.text
test_ratings_meta = test.find('div', class_ = 'inline-block ratings-metascore')

test_ratings_meta.span.text.strip()
test_gross = test.find_all('span', attrs = {'name':'nv'})

test_gross[1]
test_votes = test.find_all('span', attrs = {'name':'nv'})

test_gross[0].text
test_genre = test.find('span', class_ = 'genre' ).text.strip()

test_genre
movie_data = parser.find_all('div', class_ = 'lister-item-content')

len(movie_data)
movie_names = []

imdb_scores = []

metascores = []

gross_earned = []

votes = []

genres = []



for movie in movie_data: 

    name = movie.h3.a.text

    ratings_imdb = movie.find('div', class_ = 'inline-block ratings-imdb-rating')

    imdb = float(ratings_imdb.strong.text)

    ratings_meta = movie.find('div', class_ = 'inline-block ratings-metascore')

    g = movie.find_all('span', attrs = {'name':'nv'})

    user_votes = g[0].text

    movie_genre = movie.find('span', class_ = 'genre').text.strip()

    

    if len(g) == 2:

        gross = g[1].text

    else:

        gross = '$0M'

        

    if ratings_meta is None:

        metascore = 0

    else:

        metascore = int(ratings_meta.span.text.strip()) 

    

    movie_names.append(name)

    imdb_scores.append(imdb)

    metascores.append(metascore)

    gross_earned.append(gross)

    votes.append(user_votes)

    genres.append(movie_genre)

len(movie_names)
len(imdb_scores)
len(metascores)
#coverting to df 



movie_df = pd.DataFrame({'movie_name': movie_names, 'genres' : genres, 'imdb_score' : imdb_scores, 'metascore' : metascores, 'user_votes' : votes, 'gross_earned' : gross_earned})

movie_df
test_second_page = requests.get('https://www.imdb.com/search/title/?release_date=2017-01-01,2017-12-31&sort=num_votes,desc&start=51&ref_=adv_nxt')
test_second_page.status_code
test_second_content = test_second_page.text
test_parser_2 = BeautifulSoup(test_second_content, 'html.parser')
test_content = test_parser_2.find('div', class_ = 'lister-item-content')
test_content.h3.a.text
'https://www.imdb.com/search/title/?release_date=2017-01-01,2017-12-31&sort=num_votes,desc&start={0}&ref_=adv_nxt'.format(51)
numbers_list = list(range(51,349087,50))
numbers_list[0:4]
import contextlib

import time



@contextlib.contextmanager

def timer():

    '''Calculate time it takes for process to complete

    

    Args:

      None

      

    Yields:

       float : the time in minutes for process to run

    '''

     

    start_time = time.time()

    

    yield

    

    end_time = time.time()

    elapsed_time = (end_time - start_time) / 60

    print('The time to complete is {:.2f} minutes.'.format(elapsed_time))

    
from random import randint

from time import sleep

from IPython.core.display import clear_output



with timer():

    movie_names_list = []

    imdb_scores_list = []

    metascores_list = []

    gross_earned_list = []

    votes_list = []

    genres_list = []



    request_no = 1



    for num in numbers_list[0:5]: 

        url = 'https://www.imdb.com/search/title/?release_date=2017-01-01,2017-12-31&sort=num_votes,desc&start={0}&ref_=adv_prv'.format(num)

        resp = requests.get(url)

        sleep(randint(3,20))

    

        print('Request No: {0}'.format(request_no))

        print('Status code : {0}'.format(resp.status_code))

        print('_________________')

        clear_output(wait = True)

        request_no +=1

    

        data = resp.text

        pars = BeautifulSoup(data, 'html.parser')

    

        movie_list = pars.find_all('div', class_ = 'lister-item-content')

    

        for movie in movie_list:

    

            name = movie.h3.a.text

            ratings_imdb = movie.find('div', class_ = 'inline-block ratings-imdb-rating')

            imdb = float(ratings_imdb.strong.text)

            ratings_meta = movie.find('div', class_ = 'inline-block ratings-metascore')

            g = movie.find_all('span', attrs = {'name':'nv'})

            user_votes = g[0].text

            movie_genre = movie.find('span', class_ = 'genre').text.strip()

    

    

            if len(g) == 2:

                gross = g[1].text

            else:

                gross = '$0M'

        

            if ratings_meta is None:

                metascore = 0

            else:

                metascore = int(ratings_meta.span.text.strip())

    

            movie_names_list.append(name)

            imdb_scores_list.append(imdb)

            metascores_list.append(metascore)

            gross_earned_list.append(gross)

            votes_list.append(user_votes)

            genres_list.append(movie_genre)



len(movie_names_list)
len(imdb_scores_list)
temp_df = pd.DataFrame({'movie_name': movie_names_list, 'genres' : genres_list, 'imdb_score' : imdb_scores_list, 'metascore' : metascores_list, 'user_votes' : votes_list, 'gross_earned' : gross_earned_list})

temp_df.head()
temp_df.tail()
movie_df = pd.concat([movie_df,temp_df], axis = 0, ignore_index = True)
movie_df.shape
movie_df.head()
movie_df.tail()
movie_df.info()
movie_df['user_votes'] = movie_df['user_votes'].str.replace(',','')
movie_df['user_votes'].tail()
movie_df['user_votes'] = movie_df['user_votes'].astype(int)
movie_df['gross_earned'] = movie_df['gross_earned'].str.replace('$','')

movie_df['gross_earned'] = movie_df['gross_earned'].str.replace('M','')

movie_df.rename( columns = {'gross_earned' : 'gross_earned_millions'}, inplace = True)
movie_df.head()
movie_df['gross_earned_millions'] = movie_df['gross_earned_millions'].astype(float)
movie_df.info()
movie_df['imdb_standardized'] = movie_df['imdb_score'] * 10 
movie_df.head()
movie_df.tail()
movie_df.columns
movie_df = movie_df.iloc[:,[0,1,2,-1,3,4,5]]
movie_df.head()
# movie_df.to_csv('movie_data : 2017-01-01 and 2017-12-31')