# All libraries we will need

import pandas as pd # To store data as a dataframe

import requests # to get the data of an url

from bs4 import BeautifulSoup # to parse the html data and find what we want

import re # Regular expressions library, it may be useful 

print('Setup complete!')
# We need the url of the page we are gonna scrape 

url = 'https://pokemondb.net/pokedex/all'

response = requests.get(url) # Get content of page
# Parse the webpage text as html

page_html = BeautifulSoup(response.text, 'html.parser') 
tableContainer = page_html.find('table', attrs={'id':'pokedex'}) # Find table with id = pokedex

tableTbody= tableContainer.find('tbody') # Inside the table find tbody tag

rowsTb = tableTbody.find_all('tr') # Inside tbody get all tr = table rows

rowsTb[:2] # show first 2 tr, html shown
# Create arrays for every data you want to save, in this case all columns

name = []

types = []

total = []

hp = []

attack = []

defense = []

spAtk = []

spDef = []

speed = []
# for every row he have stored in rowsTb (rows in tbody)

for row in rowsTb:

    # Get all cells (all columns) of this row

    cells = row.find_all('td')  # This will contain an array with the content of all columns in this row

    

    # Append all content to the arrays

    # cells[0] is the id of the pokemon in this database, I didn't wanted it. 

    name.append(cells[1].text) # To get the text betweent the tags we use .text

    types.append(cells[2].text.split()) # We will use .split() because lots of pokemons have multiple types

    total.append(int(cells[3].text))

    hp.append(int(cells[4].text))

    attack.append(int(cells[5].text))

    defense.append(int(cells[6].text))

    spAtk.append(int(cells[7].text))

    spDef.append(int(cells[8].text))

    speed.append(int(cells[9].text))
# We will create pandas DataFrame to store the data collected in multiple arrays

pokePd = pd.DataFrame({'Name': name,

'Type': types,

'Total': total,

'HP': hp,

'Attack': attack,

'Defense': defense,

'Sp.Atk': spAtk,

'Sp.Def': spDef,

'Speed': speed

})
pokePd.head() # show first 5 entries of DataFrame
# We can also check the shape to see if the scrapping has been done correctly. 

pokePd.shape



# There seems to be some kind of problem, pokedex size is 893 and we have 1034. But I you look carefully you cant find that some id numbers are repeated. So there is no problem here.

# Use this to export DataFrame to csv

pokePd.to_csv('pokePd.csv')