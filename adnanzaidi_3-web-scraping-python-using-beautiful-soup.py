#Packages

#--Web scraping packages

from bs4 import BeautifulSoup

import requests



#Pandas/numpy for data manipulation

import pandas as pd

import numpy as np
#load URLs we want to scrape into an array

BASE_URL = [

    'https://www.reuters.com/companies/GOOGL.OQ/people',

    'https://www.reuters.com/companies/AMZN.OQ/people',

    'https://www.reuters.com/companies/AAPL.OQ/people'

]
#loading empty array for board members

board_members = []

#Loop through our URLs we loaded above

for b in BASE_URL:

    html = requests.get(b).text

    soup = BeautifulSoup(html, "html.parser")

    #identify table we want to scrape

    officer_table = soup.find('table', {"class" : "MarketsTable-officers-1Yb5u"})

    

    #try clause to skip any companies with missing/empty board member tables

   

    try:

        #loop through table, grab each of the 4 columns shown (try one of the links yourself to see the layout)

        for row in officer_table.find_all('tr'):

            cols = row.find_all('td')

            if len(cols) == 4:

               board_members.append((b, cols[0].text.strip(), cols[1].text.strip(), cols[2].text.strip(), cols[3].text.strip()))

    except: pass  

print(board_members)
officer_table
#convert output to new array, check length

board_array = np.asarray(board_members)

len(board_array)
#convert new array to dataframe

df = pd.DataFrame(board_array)
df.head()
#rename columns, check output

df.columns = ['URL', 'Name', 'Age','Year_Joined', 'Title']

df.head()
#Saving File as captured

df.to_csv('test.csv')