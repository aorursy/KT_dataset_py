# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from googlesearch import search
import requests
import bs4
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
from pandas import read_excel
import time
from tabulate import tabulate
df = pd.read_excel('/kaggle/input/listofdiseases/list-of-diseases.xlsx')
df.shape
size = 110
list_of_df = (df.loc[i:i+size-1,:] for i in range(0, len(df),size))
i = 0
for lst in list_of_df:
    print(lst.count())
    lst.to_csv('dieases'+str(i)+".csv")
    i +=1
df0 = pd.read_csv('dieases0.csv' , names = column)
df0.shape
def google_results(query):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.122 Safari/537.36'
    }
    r = requests.get('https://www.google.com/search?q=' + query ,headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')
    div = soup.find('div', id='result-stats')
    return int(''.join(re.findall(r'\d+', div.text.split()[1])))

new_data = []
i=1
for x in df0['Diseases Names']:
    if(i % 40 == 0):
        time.sleep(20)
    GH = google_results(x)
    new_data.append([x,GH])
    print(tabulate([[i,x,GH]]))
    #print(x," ",GH, " -->", i)
    i = i+1
newdf = pd.DataFrame(
new_data, columns=['Name', 'Google hits'])
newdf['Probability'] = newdf['Google hits']/30000000000000
newdf['Beta'] = -np.log2(newdf['Probability'])
newdf.to_csv('new_list_diseases.csv')
newdf
minbeta = newdf.min()
print(minbeta.Name)
print(minbeta.Beta)
maxbeta = newdf.max()
print(maxbeta.Name)
print(maxbeta.Beta)
from urllib.request import urlopen

request = urlopen('http://browser.ihtsdotools.org/api/v2/snomed/en-edition/v20170731/concepts/22298006/parents?form=inferred')

response_body = urlopen(request).read()
print(response_body)
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
    url = baseUrl + '/browser/' + edition + '/' + version + '/concepts?term=' + quote(searchTerm) + '&activeFilter=true&offset=0&limit=50'
    response = urlopen(url).read()
    data = json.loads(response.decode('utf-8'))

    print (data['total'])
    print (data)

#Prints number of descriptions containing the search term with a specific semantic tag
def getDescriptionsByStringFromProcedure(searchTerm, semanticTag):
    url = baseUrl + '/browser/' + edition + '/' + version + '/descriptions?term=' + quote(searchTerm) + '&conceptActive=true&semanticTag=' + quote(semanticTag) + '&groupByConcept=false&searchMode=STANDARD&offset=0&limit=50'
    response = urlopen(url).read()
    data = json.loads(response.decode('utf-8'))

    print (data['totalElements'])

getConceptsByString('heart attack')
