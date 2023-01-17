# All libraries we will need

import pandas as pd # To store data as a dataframe

import requests # to get the data of an url
# We need the url of the page we are gonna scrape 

url = 'https://pokemondb.net/pokedex/all'

response = requests.get(url) # Get content of page
pd.read_html(url)
tab = pd.read_html(response.text) # Tab is a list, so take that into account

tabDF = tab[0]

tabDF.head()
# Use this to export DataFrame to csv

tabDF.to_csv('pokedex.csv')