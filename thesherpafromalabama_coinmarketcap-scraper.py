# Imports and initializations

import requests

from bs4 import BeautifulSoup

import re

import pandas as pd

import datetime





url = 'https://coinmarketcap.com/rankings/exchanges/liquidity/'



# initialize lists

names = []

liquidity = []
r = requests.get(url)

c = r.content

soup = BeautifulSoup(c,'html.parser')

length = len(soup.find_all("tr", {"class":"cmc-table-row"}))



# Loop through and get names and liquidity 

for i in range(length):

    

    # Get name here

    names.append(soup.find_all("tr", {"class":"cmc-table-row"})[i].find("a").string)

    

    # Get liquidity metrics here

    liquidity.append(soup.find_all("td", {"class":"cmc-table__cell cmc-table__cell--sortable cmc-table__cell--right cmc-table__cell--sort-by__effective-liquidity-24-h"})[i].find("a").string)



# CLose connections 

r.close()



# Write to a pandas DF

df = pd.DataFrame(list(zip(names, liquidity)), columns =['Names', 'CMC Liquidity']) 



# Get date and hour data

date = str(datetime.datetime.now().date())

hour = str(datetime.datetime.now().hour)



# Save as a csv

df.to_csv("liquidity_data_{}_{}.csv".format(date, hour), index = False)

print ("Saved liquidity data at {}".format(str(datetime.datetime.now().time())))
