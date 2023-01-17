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
from bs4 import BeautifulSoup

import requests



url='https://www.airlinequality.com/review-pages/a-z-airline-reviews/'

response = requests.get(url)

page = response.text



soup = BeautifulSoup(page,"lxml")

url_tail=[]

airline_data=[]

airlines_name=[]



for group in soup.find_all('div', class_="a_z_col_group"):

    for each in group.find_all('li'):

        url_tail.append(each.find('a').get('href'))

        airlines_name.append(each.text)

        

base='https://www.airlinequality.com/'



for url in url_tail:

    responses = requests.get(base+url)

    pages = responses.text



    dish = BeautifulSoup(pages,"lxml")

    table=dish.find('table')

    airline_dict={}

    airline_name=airlines_name[url_tail.index(url)]

    airline_dict['Airline_Name']=airline_name

    for row in table.find_all('tr'):

        name,stars = row.find_all('td')

        airline_dict[name.text.replace(' ','_')]=int(len(stars.find_all(class_='star fill')))

    airline_dict['Review_Count']=int(dish.find('div', class_="review-count").find('span').text.strip())

    airline_dict['Rating']=int(dish.find('div', class_="rating-10 rating-large").find('span').text.strip())

    airline_data.append(airline_dict)

    

import pandas as pd

data=pd.DataFrame(airline_data)

data.set_index('Airline_Name',inplace=True)



data.to_csv('data.csv')