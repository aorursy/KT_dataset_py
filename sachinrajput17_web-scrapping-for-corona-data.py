#import important libraries for our work

import requests 

from bs4 import BeautifulSoup 

from tabulate import tabulate 

import os 

import numpy as np 

import pandas as pd

import datetime
today=datetime.date.today().strftime("%m-%d-%Y")

data_date=datetime.date.today()-datetime.timedelta(days=1)

print("Today is {}".format(today))

data_date=data_date.strftime("%m-%d-%Y")

url= 'https://www.worldometers.info/coronavirus/'
# get web data

req = requests.get(url)

response = req.content

# parse web data

soup = BeautifulSoup(response, "html.parser")

soup
# find the table

#table is in the last of the page



thead= soup.find_all('thead')[-1]

print(thead)
# get all rows in thead

head = thead.find_all('tr')

head
# get the table data content

tbody = soup.find_all('tbody')[0]

tbody
body = tbody.find_all('tr')

body
# get the table contents



# container for  column title

head_rows = []





# loop through the head and append each row to head

for tr in head:

    td = tr.find_all(['th', 'td'])

    row = [i.text for i in td]

    head_rows.append(row)

print(head_rows[0])
# container for contents

body_rows = []



# loop through the body and append each row to body

for tr in body:

    td = tr.find_all(['th', 'td'])

    row = [i.text for i in td]

    body_rows.append(row)

print(body_rows)
df_bs = pd.DataFrame(body_rows[:len(body_rows)-6],columns=head_rows[0]) 

df_bs.head()
# continentdata

cols=['Continent','TotalCases', 'NewCases', 'TotalDeaths', 'NewDeaths', 'TotalRecovered',

       'NewRecovered', 'ActiveCases', 'Serious,Critical', ]



continent_data = df_bs.iloc[:8, :-3].reset_index(drop=True)





# drop unwanted columns

continent_data = continent_data.drop('#', axis=1)

#rearrange Columns Sequence

continent_data = continent_data[cols]

continent_data['Continent'].loc[6]="Not Assigned"

continent_data['Continent'].loc[7]="World"





continent_data
# drop first 8 nrows

world_data = df_bs.iloc[8:, :-3].reset_index(drop=True)



# drop unwanted columns

world_data = world_data.drop('#', axis=1)

world_data.rename(columns={'Country,Other':"Country",'Serious,Critical':'Serious','Tests/\n1M pop\n':'Tests/1M pop'},inplace= True)



# first few rows

world_data.head()
#save the data

world_data.to_csv(str(data_date)+"world_data.csv",index = False)

world_data.head()
# check the saved data 

data=pd.read_csv(str(data_date)+"world_data.csv")

data