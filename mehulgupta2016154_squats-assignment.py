# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import requests

from bs4 import BeautifulSoup as bs

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#web page url to be extracted

url = "http://vxvault.net/ViriList.php"

def scrapper(url):

    '''Extracting Web Page using requests & BeautifulSoup '''

    response = requests.get(url)

    soup = bs(response.text,"html.parser")

    return soup

print(scrapper.__doc__)

soup=scrapper(url)
print(soup)
def extractor(soup):

        '''For storing data from web page to new dataframe'''

        table = soup.find_all('table')[0] #Extracting First Table encountered

        new_table = pd.DataFrame(columns=range(0,5) , index=[0]) #  New DataFrame to store data

        row_marker = 0

        for row in table.find_all('tr'):#extracting every row

             column_marker = 0

             columns = row.find_all('td')# extracting every element in each row

             for column in columns:

                    new_table.iat[row_marker,column_marker] = column.get_text() # storing data in new dataframe one by one

                    column_marker += 1

        return new_table

print(extractor.__doc__)

new_table=extractor(soup)
new_table
def saving_data(new_table):

        '''Saving extracted data to csv file'''  

        new_table.columns=['Date','URL','MD5','IP','Tools']

        new_table.to_csv('scrapped_data.csv')

print(saving_data.__doc__)

saving_data(new_table)