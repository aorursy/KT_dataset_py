# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

import requests



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



my_key = '8w4uVGD8Vuy2gHiAACtQsiUDr1oWnOcj'
# Code for set-up. Retrieviews raw data #



url = 'https://api.nytimes.com/svc/movies/v2/reviews/{picks}.json?api-key='+my_key

r = requests.get(url)

data = r.json()



data
# the following code retrievies the most recently written 240 movie reviews # 



all_reviews = []

offset = 0 # set offset value to 0 to start retrieving movie reviews as written in chronological order

while offset in range(0,240) : # while the offset value is between 0~160, perform the following steps

    url = 'https://api.nytimes.com/svc/movies/v2/reviews/{picks}.json?offset='+str(offset)+'&api-key='+my_key

    offset = offset + 20 # increment the offset value by 20 until it reaches 160

    r = requests.get(url) # get data from the updated url (with updated values)

    data = r.json()

    # for each reviewed movie in the 'results' dictionary, append to empty list

    for review in data['results']: 

        all_reviews.append(review)



reviews_df = pd.DataFrame(all_reviews)



reviews_df
reviews_df.to_csv('reviews.csv') # export above dataframe as CSV file
# code below is for smaller sample of data to be used for testing the model # 

test_data = []

offset = 240 # set offset value to 160 to start retrieving movie reviews from place last left off from above code

while offset in range(240,300) : # while the offset value is between 160~200, perform the following steps

    url = 'https://api.nytimes.com/svc/movies/v2/reviews/{picks}.json?offset='+str(offset)+'&api-key='+my_key

    offset = offset + 20 # increment the offset value by 20 until it reaches 160

    r = requests.get(url) # get data from the updated url (with updated values)

    test = r.json()

    # for each reviewed movie in the 'results' dictionary, append to empty list

    for review in test['results']: 

        test_data.append(review)



test_df = pd.DataFrame(test_data)



test_df
test_df.to_csv('test_data.csv') # export above dataframe as CSV file