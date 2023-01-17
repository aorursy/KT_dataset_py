import requests

from bs4 import BeautifulSoup

import pandas as pd



url = "https://in.bookmyshow.com/national-capital-region-ncr/movies/comingsoon"

headers = {

    'User-Agent': "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36"

    }

response = requests.request("GET", url, headers=headers)

data = BeautifulSoup(response.text, 'html.parser')

print(data)
movie_data = data.find_all('div', attrs={'class', 'cards cs-card-regular'})

print(movie_data)
print(len(movie_data))
for movie in movie_data:

    print(movie)
for movie in movie_data:

   movie_name = movie.find('h4')

   movie_date = movie.find('span', attrs ={'class', 'day'})

   movie_language = movie.find('li', attrs ={'class', '__language'}) 

   print(movie_name.text, movie_date.text, movie_language.text)
file = []



for movie in movie_data:

    movie_details = {}

    movie_name = movie.find('h4')

    movie_date = movie.find('span', attrs ={'class', 'day'})

    movie_language = movie.find('li', attrs ={'class', '__language'})

    movie_details['movie_name']=movie_name.text

    movie_details['movie_date']=movie_date.text

    movie_details['movie_language']=movie_language.text

    file.append(movie_details)

dataframe = pd.DataFrame.from_dict(file)

dataframe.to_csv('movie_data.csv', index=False)