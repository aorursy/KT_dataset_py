# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import re

import requests

from bs4 import BeautifulSoup as Soup
url = "https://sofifa.com/players?offset="

column = ['ID','picture','Flag','Name','Age','Position','Overall','Potential',

          'Team_Image','Team','Value','Wage','Total_Point']

FIFAdata = pd.DataFrame(columns = column)
for offset in range(0,300):

    url = url + str(offset*61)

    p_html = requests.get(url)

    p_soup = p_html.text

    data = Soup(p_soup,'html.parser')

    table = data.find('tbody')

    for i in table.findAll('tr'):    

        td = i.findAll('td')

        picture = td[0].find('img').get('data-src')

        ID = td[0].find('img').get('id')

        flag = td[1].find('img').get('data-src')

        Name = td[1].findAll('a')[1].text

        Age = td[2].text.split()

        Position = td[1].findAll('a')[2].text

        Overall = td[3].find('span').text

        Potential = td[4].find('span').text

        Team_image = td[5].find('img').get('data-src')

        Team = td[5].find('a').text

        Value = td[6].text.strip()

        Wage = td[7].text.strip()

        Total_Point = td[8].text.strip()

        player_data = pd.DataFrame([[ID,picture,flag,Name,Age,Position,Overall,Potential,

                                 Team_image,Team,Value,Wage,Total_Point]])

        #print(player_data)

        player_data.columns = column

        FIFAdata = FIFAdata.append(player_data, ignore_index = True)
print(FIFAdata.head()) 



FIFAdata.to_csv('FIFAdatas.csv', index = None)