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
#-----------------------------------------------------------------------------------------

#

# Program name : process_news.py

# Program Created Date: 31 December 2019

# Program Amanded Date: 02 Jan 2020

# Program Author: Adam McMurchie

#

# Program Description:  This program iterrogates news.api, pulls data and consolodates in a re-occuring append fasion. The aim to 

#                       Overtime build a portfolio of news data from various sources. Various archival, and time delay considerations 

#                       have been baked into the code (explained in program flow).

#

# Input Files:     Data from news.api which is whatever was present 15 minutes ago. 

# Output Files:    Json files in data folder, output.csv full list in data folder

#

# Program Flow :  **to be updated** Run job three times per day (at least)

#                  Gets current date

#                  Checks current date data folder (create if not exist)

#                  compares news array pulled from news.api to written files

#                  If news array has items that are not in written files

#                  Regenerate news array with missing items

#                  call news api, get news for updated news array

#                  Save to json

#                  Also save all files to csv (can overwrite)

#                  Saves crossing any wires or putting the wrong list in the wrong market

#

#-----------------------------------------------------------------------------------------





# IMPORT STATEMENTS 

import json

import os

from datetime import date

from newsapi import NewsApiClient

import pandas as pd



# IMPORT API KEYS (YOU WILL NEED TO REGISTER AT NEWSAPI.ORG)

f = open("../your/key/location/api.txt", "r")

keys = f.read()

f.close()

ACCESS_TOKEN = keys



#-------------------------------------------------------------------------------------------

#   PARMS SECTION 

#-------------------------------------------------------------------------------------------



# READING IN 

NewsFileNameArray = []                         # A LIST OF FILENAMES 

data = []                                      # ALL NEWS DATA FROM LOCAL NEWS JSON FILES



# REQUESTING NEW DATA

newsapi = NewsApiClient(api_key=ACCESS_TOKEN)  # INITIALISING newsapi object

news_keyname_array = ['bbc-news', 'abc-news','cnn','fox-news','independent','mirror','metro','daily-mail', 'Theguardian.com' , 'Sky.com', 'the-new-york-times', 'al-jazeera-english', 'reuters', 'the-hill' , 'breitbart-news', 'the-verge', 'the-huffington-post']

news_array = []





#-------------------------------------------------------------------------------------------

#   CREATE NEW FOLDER FOR TODAY IF NOT EXIST

#-------------------------------------------------------------------------------------------

# GET TODAYS DATE 

today = date.today()

print("Today's date:", today)



# CREATE A NEW DIRECTORY FOR TODAY IF NOT EXIST

if os.path.isdir("data/" + str(today)):

    print('Dir Already exists')

else:

    os.mkdir("data/" + str(today))



print('')
#-------------------------------------------------------------------------------------------

#   REQUESTING MORE NEWS 

#-------------------------------------------------------------------------------------------



# Init API and save to news_array

# WRITE TO JSON

print('PULLING NEWS HEADLINES - PLEASE WAIT .... ')

print('')

for item in news_keyname_array:

    print('processing ' + str(item + ' headlines'))

    news_key = item

    json_item = newsapi.get_top_headlines(sources=news_key)

    if json_item['totalResults'] == 0:

        print("Request for the " + str(item) + " news source is empty, skipping")

        print('')

        continue

    news_array.append(json_item)

    print('COMPLETE - appending to array .......')

    print('')

#-------------------------------------------------------------------------------------------

#   BUILDING REPORT 

#-------------------------------------------------------------------------------------------



# BUILD A PANDAS DATA FRAME 

df = pd.DataFrame(columns=['source','author','title','description','url', 'requested_date','publishedAt','content'])





# Iterate through DATA array and write to csv

print('iterating through ')

x = 0 

for news_outlet in range (0, len(news_array)):

    for article_number in range (0, len(news_array[news_outlet]['articles'])):

        source         = news_array[news_outlet]['articles'][article_number]['source']['name']

        author         = news_array[news_outlet]['articles'][article_number]['author']

        title          = news_array[news_outlet]['articles'][article_number]['title']

        description    = news_array[news_outlet]['articles'][article_number]['description']

        url            = news_array[news_outlet]['articles'][article_number]['url']

        publishedAt    = news_array[news_outlet]['articles'][article_number]['publishedAt']

        requested_date = today

        content        = news_array[news_outlet]['articles'][article_number]['content']

        df = df.append([{ 'source': source, 'author': author, 'title': title, 'description': description, 'url':url, 'publishedAt': publishedAt, 'requested_date': requested_date, 'content': content    }])

        x = x + 1 



print('PROCESSING COMPLETE')

print('number of articles processed are : ' + str(x))



# imported is old data

# df is the new data

# combined is merged 





try:

    f = open("data/" + str(today) + "/" + 'output.csv')

    print('File exists')

except IOError:

    print("File not accessible, saving current df to file")

    df.to_csv("data/" + str(today) + "/" + 'output.csv')



finally:

    f.close()



print('importing saved data')

imported = pd.read_csv("data/" + str(today) + "/" + 'output.csv', index_col=0)

print('Appending old and new feeds')

combined = imported.append(df)

print('')

print('dropping duplicates')

combined = combined.drop_duplicates(subset="title", keep='first')

print('')

print('describing new data')

print(df.shape)

print('')

print('describing old data')

print(imported.shape)

print('describing merged data with dropped duplicates')

print(combined.shape)



        

combined.to_csv("data/" + str(today) + "/" + 'output.csv')