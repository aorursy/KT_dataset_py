from pandas import DataFrame
from bs4 import BeautifulSoup as BS
import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup
import requests
import os
import io
import urllib.request
import numpy as np
import requests

link="https://www.usagold.com/reference/prices/goldhistory.php"

res=requests.get(link).text


url = 'https://www.usagold.com/reference/prices/goldhistory.php'

agent = {"User-Agent":'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}

page = requests.get(url, headers=agent)

soup=BS(page.content, 'lxml')


data = soup.find_all('td',{'class':'text'})
#assembling the dates with the corresponding prices 

#starting from jan 2020 until july 2020 or up to the time you are reading this 

date=[]

price=[]

k = False
for d in data:
    k = not k
    if not k:
        price.append(d.text)
        
j = False
for d in data:
    j = not j
    if  j:
        date.append(d.text)
        
#create a dataframe with 2 columns from the previous lists: 1) dates  2) prices 

data_tuples = list(zip(date,price))

gold_df=pd.DataFrame(data_tuples, columns=['date','price'])

#preparing the data by converting the date from string object to datetime pandas object

gold_df['date']= pd.to_datetime(gold_df['date'])

gold_df.sort_values(by=['date'], inplace=True)

gold_df=gold_df.reset_index()

gold_df=gold_df.drop(['index'], axis=1)
gold_df.tail()
# This link is for a github repo in which the US daily covid19 new cases reports were found in csv format

url="https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"

s=requests.get(url).content

covid_df=pd.read_csv(io.StringIO(s.decode('utf-8')))


# setting up a US new covid cases dataFrame in the proper format

covid_usa=covid_df.loc[covid_df['location']=='United States']

covid_usa= covid_usa[['date','new_cases']]

covid_usa=covid_usa.set_index('date')

covid_usa=covid_usa.reset_index()

covid_usa['date']= pd.to_datetime(covid_usa['date']) 

covid_usa.tail()


#Merging Gold prices df with US new cases df

US_graph_df=pd.merge_asof(gold_df,covid_usa, on='date', direction='backward')

US_graph_df['price'] = pd.to_numeric(US_graph_df['price'],errors='coerce')
US_graph_df.tail()
#create figure and axis objects with subplots()

fig,ax = plt.subplots(figsize=(26,12))

# make a plot  
ax.plot(US_graph_df.date, US_graph_df['price'], color="red", marker="o")

# set x-axis label
ax.set_xlabel("Date",fontsize=32)
# set y-axis label
ax.set_ylabel("Gold Price",color="red",fontsize=32)

# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object

ax2.plot(US_graph_df.date, US_graph_df["new_cases"],color="blue",marker="o")

plt.ylim(-1000,68000)

ax2.set_ylabel("New Cases",color="blue",fontsize=32)

plt.show()
