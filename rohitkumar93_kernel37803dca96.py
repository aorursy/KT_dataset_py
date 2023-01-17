import wikipedia
import requests

website_url = requests.get('https://en.wikipedia.org/wiki/List_of_institutions_of_higher_education_in_Bangalore').text
from bs4 import BeautifulSoup
soup = BeautifulSoup(website_url,'lxml')
# print(soup.prettify())
My_table = soup.find('table',{'class':'wikitable sortable'})
My_table
links = My_table.findAll('a')
links
Countries = []
for link in links:
    Countries.append(link.get('title'))
    
print(Countries)

import wikipedia
import requests
from bs4 import BeautifulSoup
import time
import numpy as np
import pandas as pd

# first pull the HTML from the page that links to all of the pages with the links.
# in this case, this page gives the links list pages of sci-fi films by decade.
# just go to https://en.wikipedia.org/wiki/Lists_of_science_fiction_films
# to see what I'm pulling from.
html = requests.get('https://en.wikipedia.org/wiki/List_of_institutions_of_higher_education_in_Bangalore').text
# print(html)
#turn the HTML into a beautiful soup text object
b = BeautifulSoup(html, 'lxml')

# print(b.find_all(name = 'li'))
# create an mpty list where those links will go.
links = []
titles = []


for i in b.find_all(name = 'li'):
    # pull the actual link for each one
    for link in i.find_all('a', href=True, title=True):
        links.append(link['href'])
    for title in i.find_all('a', href=True, title=True):
        titles.append(title['title']) 



# for i in b.find_all(name = 'li'):
#     # pull the actual link for each one
#     for title in i.find_all('a', href=True, title=True):
#         title.append(title['title'])        
        
        
decade_links = ['https://en.wikipedia.org' + i for i in links]
decade_title = [i for i in titles]      
# print(decade_title)

result = pd.DataFrame()
result['Title'] = decade_title
result['Link'] = decade_links


result



# html1 = requests.get(i in decade_links).text


details = list()
for linkers in decade_links:
    html1 = requests.get(decade_links[0]).text
    c = BeautifulSoup(html1, 'lxml')
    details.append(c.select(".mw-parser-output p:nth-of-type(3)"))

# # print(details)

# pd.DataFrame(details)


# final_result = result.append(details, ignore_index=True)
# print(final_result)


# # print(b.find_all(name = 'li'))
# # create an mpty list where those links will go.
# links = []
# titles = []
type(details)