# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import requests #For Sending Request to the Webpage

from bs4 import BeautifulSoup as Soup #For Web Scraping
link = 'https://www.worldometers.info/coronavirus/#countries'
column = ['Country','Total Cases','New Cases','Total Death','New Death','Total Recovered',

          'Active Cases','Serious Critical', 'Case Per Million', 'Death Per Million','Total Tests',

          'Test Per Million']
Virus = pd.DataFrame(columns = column)
url = requests.get(link)

url_html = url.text

page = Soup(url_html, "html.parser")
table = page.find('tbody')

for i in table.findAll('tr'):

    td = i.findAll('td')

    Country = td[0].text

    TCases  = td[1].text

    NCases  = td[2].text

    TDeath  = td[3].text

    NDeath  = td[4].text

    TRecover = td[5].text

    ACases = td[6].text

    SCritical = td[7].text

    CasePM = td[8].text

    DeathPM = td[9].text

    Ttests = td[10].text

    TestPM = td[11].text

    Virus_data = pd.DataFrame([[Country,TCases,NCases,TDeath,NDeath,TRecover,ACases,SCritical,

                                CasePM,DeathPM,Ttests,TestPM]])

    Virus_data.columns = column

    Virus = Virus.append(Virus_data, ignore_index = True)

    

print(Virus)
Virus = Virus.set_index('Country')

print(Virus.head(10))
print(Virus)
Virus.to_csv('CoronaVirus27-04.csv')  