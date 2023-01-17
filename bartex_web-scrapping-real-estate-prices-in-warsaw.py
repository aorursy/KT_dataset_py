import requests

from bs4 import BeautifulSoup

import pandas as pd

import numpy as np

import datetime
url = "https://www.olx.pl/nieruchomosci/mieszkania/sprzedaz/warszawa/?search%5Bfilter_float_price_per_m%3Afrom%5D=4000"

 

# Getting the webpage, creating a Response object.

response = requests.get(url)



# Extracting the source code of the page.

data = response.text

 

# Passing the source code to BeautifulSoup to create a BeautifulSoup object for it.

soup = BeautifulSoup(data, 'lxml')

 

# Extracting all the <a> tags into a list.

#tags = soup.find_all('a')

 

# Extracting URLs from the attribute href in the <a> tags.

#for tag in tags:

 #   print(tag.get('href'))
#urls_list = [tag.get('href') for tag in tags]

#urls_set = set(urls_list); urls_set 
# Checking if we have a right class for getting address for apartments

apartments = soup.findAll('a', {'class': 'thumb vtop inlblk rel tdnone linkWithHash scale4 detailsLink'}); apartments
# We need to define how to find the next page on ads site

next_page = soup.find('span', {'class': 'fbold next abs large'}).a.get('href')

next_page
# define a function which will fetch us a links to the apartments

def get_apartments(soup):

    return [apartment.get('href') for apartment in soup.findAll('a', {'class': 'thumb vtop inlblk rel tdnone linkWithHash scale4 detailsLink'})]
get_apartments(soup)
# Limit of pages we want to explore

limit = 10

# Starting link

url = "https://www.olx.pl/nieruchomosci/mieszkania/sprzedaz/warszawa/?search%5Bfilter_float_price_per_m%3Afrom%5D=4000"



apartments_urls = set() # set is important in order not to have the same apartment twice

for i in range(limit):

    # Getting the webpage, creating a Response object.

    response = requests.get(url)

    data = response.text

    soup = BeautifulSoup(data, 'lxml')

    

    new_apartments = get_apartments(soup)

    print(f"Discovered {len(new_apartments)} apartments")

    apartments_urls = apartments_urls.union(new_apartments)

    url = soup.find('span', {'class': 'fbold next abs large'}).a.get('href')



len(apartments_urls)
# Let's check out how we can extract data from a single ad



#url = 'https://www.olx.pl/oferta/bezposrednia-sprzedaz-mieszkanie-w-dzielnicy-wola-CID3-IDDQ9z8.html#c58c991a6e'

url = 'https://www.olx.pl/oferta/mieszkanie-2-pok-ursynow-besposriednio-blizko-metro-imielin-CID3-IDDQ2zp.html#d67bd408ad'

response = requests.get(url)

data = response.text

bs = BeautifulSoup(data, 'lxml')
# left side of the table

bs.find_all('table', {'class': 'item'})[0].th.get_text()
# right side of the table

bs.find_all('table', {'class': 'item'})[0].strong.get_text(strip=True)
a = bs.find_all('div',{'class': 'clr lheight20 large'})[0].get_text(); print(a)

# final loop - just printing



features = []



for element in bs.find_all('table', {'class': 'item'}):

    print(element.th.get_text() + ' ' + element.strong.get_text(strip=True))

print('Lokalizacja ' + bs.find_all('address')[0].p.get_text())

print(bs.find_all('div', {'class': 'offer-titlebox__details'})[0].small.get_text())
# final loop - adding to a list

features = []



for element in bs.find_all('table', {'class': 'item'}):

    features.append(element.th.get_text() + ' ' + element.strong.get_text(strip=True))

features.append('Lokalizacja ' + bs.find_all('address')[0].p.get_text())

features.append(bs.find_all('div', {'class': 'offer-titlebox__details'})[0].small.get_text())

features
variables = ['Oferta od ', 'Cena za m² ', 'Poziom ', 'Umeblowane ', 'Rynek ',

             'Powierzchnia ', 'Liczba pokoi ', 'Lokalizacja ', 'ID ogłoszenia: ', 'Rodzaj zabudowy ', 'Opis ', 'Data ']

df = pd.DataFrame(columns=variables); df

# just trying out how to extract values

for variable in variables:

    for feature in features:

        if variable in feature:

            print(feature.split(variable,1)[1])

            
# This is how it works for a single page

for variable in variables:

    for feature in features:

        if variable in feature:

            df.loc[0,variable] = feature.split(variable,1)[1]

            break # break is important in order to save processing power when using a large dataset

df
#bs.find_all('small')[0].parent.parent

bs.find_all('div', {'class': 'offer-titlebox__details'})[0].small.get_text()
# Creating an empty dataframe



# Select which variables we gonna need

variables = ['Oferta od ', 'Cena za m² ', 'Poziom ', 'Umeblowane ', 'Rynek ',

             'Powierzchnia ', 'Liczba pokoi ', 'Lokalizacja ', 'ID ogłoszenia: ', 'Rodzaj zabudowy ', 'Opis ', 'Data ']

df = pd.DataFrame(columns=variables); df

limit = 100

i = 0



for url in apartments_urls:

    if 'olx.pl' in url:

        # Data processing

        print(url)

        response = requests.get(url)

        data = response.text

        bs = BeautifulSoup(data, 'lxml')

        features = []

        

        # Extracting

        for element in bs.find_all('table', {'class': 'item'}):

            features.append(element.th.get_text() + ' ' + element.strong.get_text(strip=True)) # Podstawowe dane z tabeli

        features.append('Lokalizacja ' + bs.find_all('address')[0].p.get_text()) # Lokalizacja

        features.append(bs.find_all('div', {'class': 'offer-titlebox__details'})[0].small.get_text()) # ID ogloszenia

        features.append('Opis '+ bs.find_all('div',{'class': 'clr lheight20 large'})[0].get_text()) # Opis ogloszenia

        features.append('Data ' + bs.find_all('div', {'class': 'offer-titlebox__details'})[0].em.get_text().split(', ')[1])

        # Adding to the dataframe

        for variable in variables:

            for feature in features:

                if variable in feature:

                    df.loc[i,variable] = feature.split(variable,1)[1]

                    break # break is important in order to save processing power when using a large dataset

        df.loc[i,'Link'] = url

    

        i += 1

        if i == limit:

            break



df
# Data cleaning 

df['Powierzchnia '] = df['Powierzchnia '].str.replace(',', '.').str.replace(' m²', '').astype(float)

df['Cena za m² '] = df['Cena za m² '].str.replace(' zł/m²', '').astype(float)

# parse Polish date into international format

dic = {' stycznia ': '.01.', ' lutego ': '.02.', ' marca ': '.03.', ' kwietnia ': '.04.', ' maja ': '.05.'} # we could continue

for i,j in dic.items():

    df['Data '] = df['Data '].str.replace(i,j)
df['Data '] = df['Data '].apply(lambda x: datetime.datetime.strptime(x, '%d.%m.%Y'))
df