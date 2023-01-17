# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

from bs4 import BeautifulSoup as Soup

import requests

column = ['Name','Rating','Position','Club','League','Nationality','Speciality','Pace','Shooting','Passing','Dribbling','Defence','Physical']

fifamobile = pd.DataFrame(columns = column)
for page in range(0,715):

   link = 'https://fifarenderz.com/20/players?page='

   url = link + str(page)

   url_get = requests.get(url)

   url_text = url_get.text

   soup = Soup(url_text, 'html.parser')



   containers = soup.findAll("div", {"class":"sr-player"})

   #print(len(containers))

   #print(Soup.prettify(containers[0]))

   container = containers[0]



   for container in containers:

       name = container.findAll("span",{"class":"name"})

       name = name[0].text



       rating = container.findAll("span",{"class":"rating"})

       rating = rating[0].text



       position = container.findAll("span",{"class":"position"})

       position = position[0].text



       club = container.findAll("span")[4]

       club = club.text



       league = container.findAll("span")[5]

       league = league.text



       nation = container.findAll("span")[6]

       nation = nation.text



       special = container.findAll("span")[7]

       special = special.text



       pace = container.findAll("span",{"class":"stat-stat"})

       pace = pace[0].text



       shot = container.findAll("span")[10]

       shot = shot.text



       passing = container.findAll("span")[12]

       passing = passing.text



       dribbling = container.findAll("span")[14]

       dribbling = dribbling.text



       defence= container.findAll("span")[16]

       defence = defence.text



       physical = container.findAll("span")[18]

       physical = physical.text



       data = pd.DataFrame([[name, rating, position, club, league, nation, special, pace, shot, passing, dribbling, defence, physical]])

       data.columns = column

       fifamobile = fifamobile.append(data, ignore_index= True)
print(fifamobile) 
fifamobile.to_csv('fifamobile.csv')  