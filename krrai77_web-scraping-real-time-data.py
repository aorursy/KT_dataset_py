#load necessary libraries

import pandas as pd

import urllib.request

from bs4 import BeautifulSoup

import requests
#URLs can be fetched using the Python module urllib.request.



link = 'https://www.mohfw.gov.in/'

page = urllib.request.urlopen(link)

soup=BeautifulSoup(page)

#print (soup.prettify) #can be used to structure the data

#get the data using Request and parse it

page = "https://www.worldometers.info/coronavirus/"

data = requests.get(page)

soup = BeautifulSoup(data.content, "html.parser")

#print (soup)
#view any required HTML tag content

print ("Page title is \n",soup.title)

#print ("Tables in the page are \n",soup.find_all('table'))
#Get all links in the page

allinks=soup.find_all("a")

for link in allinks:

    links=link.get("href")

   
table=soup.find_all('table',class_='main_table_countries')

#get required content from the table



# get the table head



thead = soup.table.find('thead')

#print(thead)



# get all the rows in table head

head = thead.find_all('tr')

#print(head)



# get the table body content



tbody = soup.table.find('tbody')

#print(tbody)



# get all the rows in table body



body = tbody.find_all('tr')

#print(body)



#iterate over the table contents and store it in list



# column title

head_column = []

# table contents

rowvalues = []



# loop through the head and append each row to head

for tr in head:

    td = tr.find_all(['th', 'td'])

    row = [i.text for i in td]

    head_column.append(row)

#print(head_column)



# loop through the body and append each row to body

for tr in body:

    td = tr.find_all(['th', 'td'])

    row = [i.text for i in td]

    rowvalues.append(row)

#print(rowvalues)
#Store the list content in a DataFrame



covidglobal = pd.DataFrame(rowvalues,columns=head_column[0])         



covidglobal.head(15)
#Cleaning the data for preprocessing



covid=covidglobal.copy()

covid.drop(['#','1 Caseevery X ppl','1 Deathevery X ppl','1 Testevery X ppl'], axis=1, inplace=True)

covid=covid[7:]

covid=covid.reset_index()

covid.drop('index',axis=1,inplace=True)

covid.rename(columns={'Country,Other':'Country'}, inplace=True)