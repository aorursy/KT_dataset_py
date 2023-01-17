# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import csv
import requests
from bs4 import BeautifulSoup
url='http://quotes.toscrape.com'
req= requests.get(url)
page=req.text
page
soup= BeautifulSoup(page,'html.parser')  #Beautification using beautiful soup.
soup
quote= soup.findAll('div',{'class':'quote'}) #Extracting only the quotes from the div class in which it is present.
quote

scrapped=[]

for i in quote:
    text= i.find('span',class_='text').text
    author=i.find('small',class_='author').text
    scrapped.append([text,author])
scrapped
with open('Quotes1.csv','w',encoding='utf-8',newline='') as file:
    writer=csv.writer(file)
    writer.writerow(["Quote","Writer"])
    
    for i in scrapped:
        writer.writerow(i)
import pandas as pd
df=pd.read_csv('Quotes1.csv')
df
!pip install fake-useragent
## importing bs4, requests, fake_useragent and csv modules
import bs4
import requests
from fake_useragent import UserAgent
import csv

## create an array with URLs
urls = ['http://quotes.toscrape.com', 'http://quotes.toscrape.com/page/2/','http://quotes.toscrape.com/page/3/',
       'http://quotes.toscrape.com/page/4/','http://quotes.toscrape.com/page/5/','http://quotes.toscrape.com/page/6/',
       'http://quotes.toscrape.com/page/7/','http://quotes.toscrape.com/page/8/','http://quotes.toscrape.com/page/9/',
       'http://quotes.toscrape.com/page/10/']

## initializing the UserAgent object
user_agent = UserAgent()

## starting the loop
for url in urls:
    ## getting the reponse from the page using get method of requests module
    page = requests.get(url, headers={"user-agent": user_agent.chrome})

    ## storing the content of the page in a variable
    html = page.content

    ## creating BeautifulSoup object
    soup = bs4.BeautifulSoup(html, "html.parser")

    ## Then parse the HTML, extract any data
    ## write it to a file
    
    quotes = soup.findAll('div', class_='quote')
    
    scraped = []
    for quote in quotes:
        text = quote.find('span', class_='text').text
        author = quote.find('small', class_='author').text
        scraped.append([text, author])
        
        with open('Quotes2.csv','a',encoding='utf-8',newline='') as file:
            writer=csv.writer(file)
            writer.writerow(["Quote","Writer"])
    
            for i in scraped:
                writer.writerow(i)
df2=pd.read_csv("Quotes2.csv")
df2
df2.shape
