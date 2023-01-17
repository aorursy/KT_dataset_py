from googlesearch import search

import requests

import bs4

from bs4 import BeautifulSoup

import re

import pandas as pd

import numpy as np

from pandas import read_excel
column = ['Diseases Names']

dfs = pd.read_excel('/kaggle/input/listofdiseases/list-of-diseases.xlsx' , names = column)
def google_results(query):

    headers = {

        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36'

    }

    r = requests.get('https://www.google.com/search?q=' + query ,headers=headers)

    soup = BeautifulSoup(r.text, 'html.parser')

    div = soup.find('div', id='result-stats')

    return int(''.join(re.findall(r'\d+', div.text.split()[1])))



def google_documentry_results(query):

    headers = {

        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36'

    }

    r = requests.get('https://www.google.com/search?q=' + query ,headers=headers)

    soup = BeautifulSoup(r.text, 'html.parser')

    div = soup.find('div', id='result-stats')

    divs = soup.find('title')

    return int(''.join(re.findall(r'\d+', div.text.split()[1])))



df= pd.DataFrame( dfs.sample(n = 50, random_state = 1) )

new_data = []

for x in df['Diseases Names']:

    GH = google_results(x)

    GDH = google_results('research document ' + x)

    new_data.append([x,GH,GDH])

newdf = pd.DataFrame(

new_data, columns=['Name', 'Google hits','Google Document hits'])

newdf
newdf['Probability'] = newdf['Google hits']/newdf['Google Document hits']

newdf['Beta'] = -np.log2(newdf['Probability'])

newdf
minbeta = newdf.min()

print(minbeta.Name)

print(minbeta.Beta)
maxbeta = newdf.max()

print(maxbeta.Name)

print(maxbeta.Beta)
newdf.to_csv('new_list_diseases.csv')
from urllib.request import urlopen

from urllib.parse import quote

import json
baseUrl = 'https://browser.ihtsdotools.org/snowstorm/snomed-ct'

edition = 'MAIN'

version = '2020-03-09'



#Prints fsn of a concept

def getConceptById(id):

    url = baseUrl + '/browser/' + edition + '/' + version + '/concepts/' + id

    response = urlopen(url).read()

    data = json.loads(response.decode('utf-8'))



    print (data['fsn']['term'])



#Prints description by id

def getDescriptionById(id):

    url = baseUrl + '/' + edition + '/' + version + '/descriptions/' + id

    response = urlopen(url).read()

    data = json.loads(response.decode('utf-8'))



    print (data['term'])



#Prints number of concepts with descriptions containing the search term

def getConceptsByString(searchTerm):

    url = baseUrl + '/browser/' + edition + '/' + version + '/concepts?term=' + quote(searchTerm) + '&activeFilter=true&offset=0&limit=50' + '/parents?form=' + quote(searchTerm)

    response = urlopen(url).read()

    data = json.loads(response.decode('utf-8'))



    print (response)





#Prints number of descriptions containing the search term with a specific semantic tag

def getDescriptionsByStringFromProcedure(searchTerm, semanticTag):

    url = baseUrl + '/browser/' + edition + '/' + version + '/descriptions?term=' + quote(searchTerm) + '&conceptActive=true&semanticTag=' + quote(semanticTag) + '&groupByConcept=false&searchMode=STANDARD&offset=0&limit=50'

    response = urlopen(url).read()

    data = json.loads(response.decode('utf-8'))



    print (data['totalElements'])

#getConceptsByString('HIV')

getConceptsById(116680003)
getConceptById('116680003')

