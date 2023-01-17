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
#Web Access and parsing

import urllib.request, urllib.parse

from bs4 import BeautifulSoup



# Importing Data

netflix_data = pd.read_csv("../input/netflix-shows/netflix_titles.csv")

netflix_data.head() #Displaying first few entries

#np.where(pd.isnull(netflix_data.release_year)) #checking for empty values for release years, None found, **might implement later just in case**
#IMDB Access/Search and Parse HTML to grab title_ID

def grab_ID(title,year,media_type):

    encoding = 'utf-8' #standard internet encoding

    find_url = 'https://www.imdb.com/find' #base search URL for IMDB

    values = {'q':title} # What we are searching for

    query = urllib.parse.urlencode(values).encode(encoding) #parsing the query and encoding correctly for web to understand it

    req = urllib.request.Request(find_url,query) #request: the "link" you can paste in the search bar

    resp = urllib.request.urlopen(req) #sending the request and recording the response

    html = BeautifulSoup(resp,"html") #using Beautiful Soup 4 to process the HTML response

    #Finding top 3 results through <td> tags with result text class in the HTML response. <td> is a table cell in HTML

    search = html.findAll("td", {"class": "result_text"})[:3]

    if len(search) > 0:

        for item in search: 

            if  str(item.text).find(str(year)) > 0:

                if str(media_type) == "TV Show":

                    if str(item.text).find("Series") > 0:

                        title_names = str(item.text)

                        title_IDs = str(item.a["href"])

                        #confidence = 

                        break

                    else:

                        #confidence = 

                        title_names = str(item.text)

                        title_IDs = str(item.a["href"])

                        break

                else:

                    title_names = str(item.text)

                    title_IDs = str(item.a["href"])

                    #confidence =

                    break



            else:

                if str(media_type) == "TV Show":

                    if str(item.text).find("Series") > 0:

                        title_names = str(item.text)

                        title_IDs = str(item.a["href"])

                        #confidence =

                        break

                    else:

                        title_names = str(item.text)

                        title_IDs = str(item.a["href"])

                        #confidence =

                        break

                else:

                    title_names = "Not Found"

                    title_IDs = "Not Found"

                    #confidence =

                    break

    else:

        title_names = "Not Found"

        title_IDs = "Not Found"

        #confidence =



    return title_names, title_IDs
#Acessing IMDB and parse HTML to grab Ratings

def grab_Rating(title_ID):

    imdb_url = "https://www.imdb.com" #base imdb website

    if title_ID == "Not Found":

        rating = "Not Found"

    else:

        target_url = imdb_url + title_ID #url for title

        target_url = target_url

        resp = urllib.request.urlopen(target_url)

        html = BeautifulSoup(resp,"html") #using Beautiful Soup 4 to process the HTML responce

        search = html.findAll("div", {"class": "ratingValue"})

        if len(search) > 0:

            for item in search: #Grabing rating text

                rating = str(item.strong['title'])

        else:

            rating = "Not Found"



    return rating

IMDB_rating = []

IMDB_titleID = []

IMDB_title_name = []

NumShows = len(netflix_data.index)

for i in range(NumShows):

    title_name, title_ID = grab_ID(netflix_data.title[i],netflix_data.release_year[i],netflix_data.type[i])

    IMDB_titleID.append(title_ID)

    IMDB_title_name.append(title_name)

    rating = grab_Rating(title_ID)

    IMDB_rating.append(rating)

    if i%100 == 0:

        print("Progress is: ", i)

    #print(netflix_data.title[i], "\t\t" title_name,"\t\t" , i)
#Writing outputs

output = pd.DataFrame({'IMDB_titleID': IMDB_titleID, 'IMDB_rating': IMDB_rating, 'IMDB_title_name': IMDB_title_name})

output_name = 'IMDB_results_' + 'jan-28-2020' + '.csv'

output.head()

output.to_csv(output_name, index=False)

print("Complete!")


#Testing Cell

#loco = 290

#x = netflix_data.title[loco]

#y = netflix_data.type[loco]

#z = netflix_data.release_year[loco]

#netflix_data.loc[loco]



#print(x, z, y)

#title_name, title_ID = grab_ID(x,z, y)

#print(title_name)

#print(title_ID)

#rating = grab_Rating(title_ID)

#print(rating)

#print(len(netflix_data.index))