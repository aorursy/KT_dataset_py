#Importing necessary library

import pandas as pd

import numpy as np

import requests

from pandas_profiling import ProfileReport



import urllib.request



#import the beatiful soup functions to parse the data

from bs4 import BeautifulSoup



# Any results you write to the current directory are saved as output.

from IPython.display import display, Image



import math



display(Image(filename='/kaggle/input/web-scraping-technique.jpg'))
URL = "https://coronanepal.live/"

#query the website

page = requests.get(url = URL)



#parse the html and store in Beautiful soup format

soup = BeautifulSoup(page.text)



#find all links

all_links = soup.find_all("a")

for link in all_links:

    pass

    #print(link.get("href"))

    



#find all tables

all_tables = soup.find('table')



#Uncomment Below line to get more sense of data

#all_tables
#Generate lists

A = []

B = []

C = []

D = []

E = []

F = []



for row in all_tables.findAll("tr"):

    cells = row.findAll('td')

    

    #Only extract table body

    if(len(cells) == 6):

        A.append(cells[0].find(text = True))

        B.append(cells[1].find(text = True))

        C.append(cells[2].find(text = True))

        D.append(cells[3].find(text = True))

        E.append(cells[4].find(text = True))

        F.append(cells[5].find(text = True))
#Creating Empty DataFrame

df = pd.DataFrame()



#Creating columns and stroing value

df['जिल्ला'] = A

df['District'] = B

df['Confirmed Cases'] = C

df['कुल संक्रमित'] = D

df['जम्मा मृत्यु'] = E

df['निको भएको'] = F



#Defining datatype for each column

df = df.astype({'जिल्ला':str,'District':str,'कुल संक्रमित':int,'जम्मा मृत्यु':int, 'निको भएको':int})



#Displaying the data

from IPython.display import display, HTML

display(HTML(df.to_html()))
URL = "https://pomber.github.io/covid19/timeseries.json"



# sending get request and saving the response as response object 

r = requests.get(url = URL) 



# extracting data in json format 

data = r.json() 



#It contains all countries data.Extracting data of only Nepal

nepal_data = data['Nepal']



#Uncomment below line to get more sense of Data

#print(nepal_data)
date_list = []

confirmed_list = []

recovered_list = []

death_list = []



for i in range(len(nepal_data)):

    date_list.append(nepal_data[i]['date'])

    confirmed_list.append(nepal_data[i]['confirmed'])

    recovered_list.append(nepal_data[i]['recovered'])

    death_list.append(nepal_data[i]['deaths'])
#Creating empty DataFrame

nepal_data_frame = pd.DataFrame()  



#Putting data in column from list

nepal_data_frame['Date'] = date_list

nepal_data_frame['Confirmed'] = confirmed_list

nepal_data_frame['Recovered'] = recovered_list

nepal_data_frame['Death'] = death_list







nepal_data_frame = nepal_data_frame.astype({'Confirmed':int, "Recovered":int})



#Displaying the data

from IPython.display import display, HTML

display(HTML(nepal_data_frame.to_html()))
#install library

!pip install COVID19Py
import COVID19Py



covid19 = COVID19Py.COVID19()
data = covid19.getAll()



#Uncomment below line to get sense of data

#print(data)
latest = covid19.getLatest()

print(latest)