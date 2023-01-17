# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import requests
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
enter = 1
cols = {}
df.drop(df.index, inplace=True)
while( enter ==1 ):
    print(" Enter release year to get movies : \n")
    year = int(input())
    response = requests.get('https://api.themoviedb.org/3/discover/movie?api_key=' + '6e5b6f5cf89a22362c5d79cc7e9b539b' + '&primary_release_year=' + str(year) + '&sort_by=revenue.desc')
    responded = response.json()
    resp = responded['results']
    title =[]
    for i in resp:
        title.append(i['title'])
    cols['Title - ' + str(year)] = title
    
    print("Do you want to continue for some other year: \n")
    print("Enter 1 for YES\n")
    print("Enter 0 for No\n")
    enter = int(input())
df = pd.DataFrame(cols)
enter2 = 1
upcoming_list = []
top_rated = []
while( enter2 ==1 ):
    
    print("For Upcoming Movies Enter 1\n")
    print("For Top Rated Movies Enter 2\n")
    
    choice2 = int(input())
    
    if(choice2 == 1):
        response = requests.get('http://api.themoviedb.org/3/movie/upcoming?api_key=' + '6e5b6f5cf89a22362c5d79cc7e9b539b')
        upcomings = response.json()
        movie_list = upcomings['results']
        for movie in movie_list:
            upcoming_list.append(movie['title'])
    print("Enter 0 if you want to exit : \n")
    enter2 = int(input())
df['Upcoming Movies'] = upcoming_list
df
import urllib.request
from PIL import Image 
url = 'https://image.tmdb.org/t/p/w185/or06FN3Dka5tukK1e9sl16pB3iy.jpg'
img = Image.open(urllib.request.urlopen(url))
print(img)
img.height
img.width
