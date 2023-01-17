import re

import requests

import pandas as pd

from bs4 import BeautifulSoup

from time import sleep

from selenium import webdriver

import matplotlib.pyplot as plt

plt.style.use('classic')

%matplotlib inline

import numpy as np

import pandas as pd

import seaborn as sns
resp = requests.get('http://www.husseinattar.com/ar/saudi-tech-startups/')

resp.status_code
html = resp.text
soup = BeautifulSoup(html, 'lxml')
soup.find_all('div', attrs={'class':'booking'})
!pip install selenium

!pip install nltk
import nltk

from nltk.tokenize import PunktSentenceTokenizer

from nltk.stem import PorterStemmer

from nltk.corpus import stopwords
driver = webdriver.Chrome(executable_path='chromedriver/chromedriver')

driver.get('http://www.husseinattar.com/ar/saudi-tech-startups/') 

sleep(10)

html = driver.page_source

# driver.find_element_by_css_selector('div.button.c_button.s_button').click()

driver.close()
soup = BeautifulSoup(html, "lxml")
names = soup.find_all('div', attrs={'class':'LinkLibraryCat LinkLibraryCat88 level0'})

names
all_names = soup.find_all('div', attrs={'class':'tatsu-row'})
all_names
stups_name = []

for hit in soup.findAll(attrs={'class' : 'track_this_link'}):

     stups_name.append(hit.contents[0].strip())
stups_name
X = soup.find_all("div", {"class" : "tatsu-column-inner"})
len(X)
X[2]
my_range=25

catg_list=[]

startup_name=[]

desc=[]

link=[]

for our_range in range(my_range):

    print('in catagory')

    print(our_range)

    print(my_range)

    for current_list in X[our_range].findAll('li'):

        print(X[our_range].find('div',{'class':'linklistcatname'}).text)

        catg_list.append(X[our_range].find('div',{'class':'linklistcatname'}).text)

        

        print(current_list.find('a').text)

        startup_name.append(current_list.find('a').text)

        

        print(current_list.find('a')['href'])

        link.append(current_list.find('a')['href'])

        

        print(current_list.find('a')['title'])

        desc.append(current_list.find('a')['title'])

        
df_startup = pd.DataFrame({'startup_name':startup_name,

              'startup_catagory':catg_list,

             'description':desc,

              'link_to_startup':link

             })
classes = soup.find_all('div', attrs={'class':'linkcatname'})
df_startup.head()
df_startup.to_csv("saudi-startups.csv")
from nltk.tag import pos_tag

from nltk.tokenize import WordPunctTokenizer
# # Lets use the stop_words argument to remove words like "and, the, a"

# cvec = CountVectorizer(stop_words='arabic')



# # Fit our vectorizer using our train data

# cvec.fit(data_train['data'])
catagories = df_startup.startup_catagory.unique()

catagories_ = df_startup.startup_catagory
len(catagories)
for i in range(len(catagories)):

    df_startup.startup_catagory.replace(to_replace =[catagories[i]], 

                     value =i, inplace=True)
df_startup.info()
df_startup.startup_catagory.value_counts()
df_startup.groupby(['startup_catagory'])['startup_catagory'].value_counts(sort=False).plot('bar')
# import time

# from selenium import webdriver

# from selenium.webdriver.common.by import By

# from selenium.webdriver.support.ui import WebDriverWait

# from selenium.webdriver.support import expected_conditions as EC

# from selenium.webdriver.common.keys import Keys



# browser = webdriver.Chrome('chromedriver/chromedriver')

# # browser = webdriver.Chrome(executable_path=r'C:/Utility/BrowserDrivers/chromedriver.exe')

# browser.get('http://www.google.com')



# # for link in df_startup.link_to_startup:

# #     browser.get(link)



# search = browser.find_element_by_name('q')

# for name in df_startup.startup_name:

#     search.send_keys(name)

#     search.send_keys(Keys.RETURN) # hit return after you enter search text

#     time.sleep(5) # sleep for 5 seconds so you can see the results

# browser.quit()
