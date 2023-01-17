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
from requests import get

url = 'https://www.imdb.com/search/name/?birth_place=India&adult=include&count=100&start=1&ref_=rlm'

response = get(url)

print(response.text[:500])

from bs4 import BeautifulSoup

html_soup = BeautifulSoup(response.text, 'html.parser')

type(html_soup)
stars_containers = html_soup.find_all('div', class_ = 'lister-item mode-detail')

print(type(stars_containers))

print(len(stars_containers))
# Picking up the first division container.

first_star=stars_containers[0]
names=[]

category=[]

images=[]

well_known_movie=[]

bio=[]



stars_containers = html_soup.find_all('div', class_ = 'lister-item mode-detail')



for container in stars_containers:

    

    #Name of celebrity

    name = container.h3.a.text.strip()

    names.append(name)

    

    #Image of celebrity

    image=container.find('img')['src']

    images.append(image)

    

    #Category of celebrirty

    categ=container.p.text.split('|')[0].strip()

    category.append(categ)

    

    #Well known movie of celebrity

    well=container.p.a.text.strip()

    well_known_movie.append(well)

    

    #bio of celebrity

    bioa=container.text.split(well)[1].strip()

    biot=list(bioa.split("."))[0]

    bio.append(biot)

    
#Now we will be scrapping multiple pages
# Create a list called pages, and populate it with the strings 

pages = [str(i) for i in range(1,4766,100)]
from time import sleep

from random import randint

from time import time

start_time = time()

requests = 0

for _ in range(5):

# A request would go here

    requests += 1

    sleep(randint(1,3))

    elapsed_time = time() - start_time

    print('Request: {}; Frequency: {} requests/s'.format(requests, requests/elapsed_time))

from IPython.core.display import clear_output

start_time = time()

requests = 0

for _ in range(5):

# A request would go here

    requests += 1

    sleep(randint(1,3))

    current_time = time()

    elapsed_time = current_time - start_time

    print('Request: {}; Frequency: {} requests/s'.format(requests, requests/elapsed_time))

clear_output(wait = True)
from warnings import warn

warn("Warning Simulation")
names=[]

category=[]

images=[]

well_known_movie=[]

bio=[]



start_time = time()

requests = 0



for page in pages:

    

    response = get('https://www.imdb.com/search/name/?birth_place=India&adult=include&count=100&start=' + page + 

    '&ref_=rlm')

    

    # Pause the loop

    sleep(randint(8,15))



    # Monitor the requests

    requests += 1

    elapsed_time = time() - start_time

    print('Request:{}; Frequency: {} requests/s'.format(requests, requests/elapsed_time))

    clear_output(wait = True)



    # Throw a warning for non-200 status codes

    if response.status_code != 200:

        warn('Request: {}; Status code: {}'.format(requests, response.status_code))



    # Break the loop if the number of requests is greater than expected

    if requests > 72:

        warn('Number of requests was greater than expected.')

        break

        

    # Parse the content of the request with BeautifulSoup

    page_html = BeautifulSoup(response.text, 'html.parser') 

    

    stars_containers = page_html.find_all('div', class_ = 'lister-item mode-detail')

    

    for container in stars_containers:

        

        if container.find('span', class_='ghost') is not None:

        

            #Name of celebrity

            name = container.h3.a.text.strip()

            names.append(name)



            #Image of celebrity

            image=container.find('img')['src']

            images.append(image)



            #Category of celebrirty

            categ=container.p.text.split('|')[0].strip()

            category.append(categ)



            #Well known movie of celebrity

            well=container.p.a.text.strip()

            well_known_movie.append(well)



            #bio of celebrity

            bioa=container.text.split(well)[1].strip()

            biot=list(bioa.split("."))[0]

            bio.append(biot)

    
import pandas as pd

Celebrity_info = pd.DataFrame({'Name': names,

'Category': category,

'Image': images,

'Well_Known_Movie': well_known_movie,

'Bio': bio

})

print(Celebrity_info.info())

Celebrity_info.head(10)
# To print in HTML format.

from IPython.core.display import HTML



# convert your links to html tags 

def path_to_image_html(path):

    return '<img src="'+ path + '" width="60" >'



pd.set_option('display.max_colwidth', -1)



HTML(Celebrity_info.to_html(escape=False ,formatters=dict(Image=path_to_image_html)))
# Let's save it.

Celebrity_info.to_csv('celeb_info_csv.csv')