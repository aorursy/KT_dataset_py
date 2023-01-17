#Import pandas to convert list to data frame



import pandas as pd

import numpy as np

import requests

from pandas_profiling import ProfileReport



import urllib.request



#import the beatiful soup functions to parse the data

from bs4 import BeautifulSoup
URL = "http://gbsnote.com/slc-result-history-nepal/"

#query the website

page = requests.get(url = URL)



#parse the html and store in Beautiful soup format

soup = BeautifulSoup(page.text)
soup.title.string
#find all links



all_links = soup.find_all("a")

for link in all_links:

    pass

    #print(link.get("href"))

#find all tables

all_tables = soup.find('table')

#print(all_tables)
#Generate lists



A = []

B = []

C = []

D = []



for row in all_tables.findAll("tr"):

    cells = row.findAll('td')

    

    #Only extract table body

    if(len(cells) == 4):

        A.append(cells[0].find(text = True))

        B.append(cells[1].find(text = True))

        C.append(cells[2].find(text = True))

        D.append(cells[3].find(text = True))
df = pd.DataFrame()



df['Year(BS)'] = A

df['Total Appeared'] = B

df['Total Passed'] = C

df['Pass Percentage'] = D

truedf = df[1:]

truedf['Pass Percentage'] = truedf['Pass Percentage'].str.replace('%', '')

truedf['Year(BS)'] = truedf['Year(BS)'].str.replace('\n', '0')
truedf.to_csv('slc.csv', encoding='utf-8', index=False)
#Read in data

features = pd.read_csv('slc.csv')
type(features['Year(BS)'])
features.mean()
#Replacing zero value of year

past = 0

value = 0

for value in (features['Year(BS)']):

    #print(past)

    if(value != 0):

        past = value

    else:

        nextval = past+1

        features.replace(value,nextval)

        break

print(nextval)
finalfea = features.replace(value,nextval)

finalfea.to_csv('slcfinal.csv', encoding='utf-8', index=False)
finalfea.head()
profile = ProfileReport(finalfea, title='Pandas Profiling Report', html={'style':{'full_width':True}})
profile.to_notebook_iframe()
finalfea.describe()
from plotly.offline import iplot, init_notebook_mode

import cufflinks as cf

cf.go_offline()

print( cf.__version__)
finalfea_process = finalfea[['Year(BS)','Total Appeared', 'Total Passed']].set_index('Year(BS)')

finalfea.iplot(x='Year(BS)', y=['Total Appeared', 'Total Passed'], kind='bar',xTitle='Year(BS)')
import matplotlib.pyplot as plt
finalfea.iplot(x= 'Year(BS)',y = 'Pass Percentage',xTitle='Year(BS)', yTitle='Pass Percentage')
finalfea.iplot(x= 'Year(BS)',y = 'Total Appeared',xTitle='Year(BS)', yTitle='Total Appeared')
finalfea.iplot(x= 'Year(BS)',y = 'Total Passed',xTitle='Year(BS)', yTitle='Total Passed')
finalfea.iplot(x= 'Year(BS)',y = ['Total Appeared','Total Passed'],xTitle='Year(BS)')