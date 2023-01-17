from urllib.request import urlopen

from bs4 import BeautifulSoup

import pandas as pd

import html5lib
#Grabbing our URL and opening it for analysis of HTML

url = "https://en.wikipedia.org/wiki/Crime_in_the_United_States"

html = urlopen(url)



#Allows us to pull HTML data for Python use

soup = BeautifulSoup(html, 'html.parser')
column_headers = [soup.findAll('th')[i].getText() for i in range (12,18)]

column_headers
row_data = soup.findAll('tr')[15:71]
type(row_data)
row_data
data_list= []

for i in range(len(row_data)):

    row=[]

    for td in row_data[i].findAll('td'):

        row.append(td.getText())

    data_list.append(row)

    

df= pd.DataFrame(data_list, columns=column_headers)



    
df.head()
df.tail()
df.info()
print(type(df["Year"][0]))

print(type(df["Rape"][0]))
df["Year"][0]
df2 = df.apply(pd.to_numeric)
df2["Year"][3]
type(df2["Year"][3])

type(df2["Rape"][7])