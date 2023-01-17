from bs4 import BeautifulSoup

import requests

import time

import random as ran

import sys

import pandas as pd

import numpy as np



from time import sleep

from random import randint
#Empty variables

Names = []
headers = {"Accept-Language": "en-US,en;q=0.5"}

page = requests.get("https://www.giantbomb.com/microtransaction/3015-199/games/?page=1", headers=headers)

soup = BeautifulSoup(page.text, 'html.parser')



#Total Number of Pages to Iterate

page_ref = soup.find('ul',{'class':'paginate js-table-paginator'}).findAll("li")

x = len(page_ref) - 2 #print(x)



#print(page_ref[x]) #print(page_ref)

page_num = int(page_ref[x].find('a').text)



#print(page_num)







#Iteration Part

pages = np.arange(1, page_num+1, 1)



for page in pages: 

    #request web pages in iterative responses

    page = requests.get("https://www.giantbomb.com/microtransaction/3015-199/games/?page=" + str(page), headers=headers)

    soup = BeautifulSoup(page.text, 'html.parser')

    vg_div = soup.find_all('li',{'class':'related-game'})

    #control the crawl rate

    #sleep(randint(2,10))

        

    for vg in vg_div:



        #name

        name = vg.find('a').find('h3').get_text()

        Names.append(name)





videogames = pd.DataFrame({

    'Title': Names})

pd.DataFrame(videogames)

#convert to csv



#videogames.to_csv('videogames_microtransactions.csv')