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
#CEPSTRAL DEMO

%reset -f

#!/usr/bin/python

# -- coding: utf-8 --



# Download Chrome driver: https://sites.google.com/a/chromium.org/chromedriver/downloads

# Add Chromedriver to PATH Variable



from selenium import webdriver

import time,os





url = 'https://www.tripadvisor.com/Restaurants/demos' #The URL from which data is to scraped

path = "/Users/arpitbhal/Desktop/chromedriver"

browser = webdriver.Chrome(path) 

browser.get(url) #Open the url in a new tab of the browser

element = browser.find_element_by_xpath('//*[@id="taplc_trip_search_home_restaurants_0"]/div[2]/div[1]/div/span/input') #Find the search bar in the page

s='Sant Cugat'

element.clear()

element.send_keys(s) 

browser.find_element_by_xpath('//*[@id="SUBMIT_RESTAURANTS"]/span[2]').click()

time.sleep(5)

#Extracting top Restaurants based on rankings and xpath

names = browser.find_elements_by_xpath("//*[contains(@class,'restaurants-list-ListCell__nameBlock')]")

links = browser.find_elements_by_xpath("//*[contains(@class,'restaurants-list-ListCell__restaurantName')]")



lists = []  #Create an empty list to store the Restaurant names and links

#Getting the top 10 Restaurants

for i in range(1,11):  

    first = {'name':names[i].text, 'link':links[i].get_attribute('href')}

    lists.append(first)



    # Connect to Mongodb

import pymongo





try:

    with open("credentials.txt", 'r') as f:

        [name,password,url]=f.read().splitlines()

        conn=pymongo.MongoClient("mongodb+srv://{}:{}@{}".format(name,password,url))

    print ("Connected successfully!!!")

except pymongo.errors.ConnectionFailure as e:

        print ("Could not connect to MongoDB: %s" % e) 



#Creating Database and collection and storing our data to it

db = conn['TripAdvisor']

collection = db['Restaurants']

collection.insert_many(lists)

print (collection.count_documents({}))  #Print the number of items stored