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
import requests

import urllib.request

import time

from bs4 import BeautifulSoup

from urllib.request import urlopen
url = "https://me.cleartrip.com/hotels/united-states/miami/"

response=requests.get(url)
content=BeautifulSoup(response.content, "html.parser")

print(content)
#to make a new connection with new link

def new_conn(link):

    response_1=requests.get(link)

    con=BeautifulSoup(response_1.content, "html.parser")

    return con
#function for providing amenities and other info.

def hotel_info(link):

    response_1=requests.get(link)

    info_con=BeautifulSoup(response_1.content, "html.parser")

    dict = {}

    for data in info_con.findAll('div', attrs={"class": "amenitiesCategory"}):

        for pdata2 in data.find_all('p'):

            btags = pdata2.find('b')

            if btags!=None:

                btags=btags.get_text().strip()

                pdata2=pdata2.get_text().strip()

                dict[btags]= pdata2    

    return dict
#all info of hotel of First-Page



content = BeautifulSoup(response.content, "html.parser")



hotel_list = []



for data in content.findAll('div', attrs={"class": "ct-hotels-card"}):

    HotelName = data.find('h2', attrs={"class": "truncate"})

    

    Address = data.find('div', attrs={"class": "area-name truncate"})

    if Address==None:

        Address=" "

    else:

        Address=Address.get_text().strip()

    

    tag = data.find('h2', attrs={"class": "truncate"})

    tags = tag.find_all('a')

    for l in tags:

        lin=l.get('href')

        link="https://me.cleartrip.com"+lin

    

    mainprice = data.find('div', attrs={"class": "main-price"})

    if mainprice==None:

        mainprice=" "

    else:

        mainprice=mainprice.get_text().strip()

         

    tax = data.find('div', attrs={"class": "tax-price"})

    if tax==None:

        tax=" "

    else:

        tax=tax.get_text().strip()

        

    reviews = data.find('span', attrs={"class": "taReviews"})

    if reviews==None:

        reviews=" "

    else:

        reviews=reviews.get_text().strip()

    

    house_info = hotel_info(link)

    

    

    

    hotelObject = {

        "Hotel_Name" : HotelName.get_text().strip(),

        "Address" : Address,

        "Hotel_URL" : link,

        "mainprice" : mainprice,

        "tax" : tax,

        "reviews" : reviews,

        "house_info" : house_info

        }  

    

    conn = new_conn(link)

    for data in conn.findAll('div', attrs={"class": "hInfo row"}):

        for data2 in data.find_all('li'):

            atags = data2.find('small')

            btags = data2.find('span')

            atags = atags.get_text().strip()

            btags = btags.get_text().strip()

            hotelObject[atags] = btags

    

    

    hotel_list.append(hotelObject) 

    

for list in hotel_list:

    print(list,"\n")
# hotel_list = []
# #all info of hotel of all pages



# def navigate_page(lin):

#     response=requests.get(lin)

#     content = BeautifulSoup(response.content, "html.parser")

    

#     for data in content.findAll('div', attrs={"class": "ct-hotels-card"}):

#         HotelName = data.find('h2', attrs={"class": "truncate"})



#         Address = data.find('div', attrs={"class": "area-name truncate"})

#         if Address==None:

#             Address=" "

#         else:

#             Address=Address.get_text().strip()



#         tag = data.find('h2', attrs={"class": "truncate"})

#         tags = tag.find_all('a')

#         for l in tags:

#             lin=l.get('href')

#             link="https://me.cleartrip.com"+lin



#         mainprice = data.find('div', attrs={"class": "main-price"})

#         if mainprice==None:

#             mainprice=" "

#         else:

#             mainprice=mainprice.get_text().strip()



#         tax = data.find('div', attrs={"class": "tax-price"})

#         if tax==None:

#             tax=" "

#         else:

#             tax=tax.get_text().strip()



#         reviews = data.find('span', attrs={"class": "taReviews"})

#         if reviews==None:

#             reviews=" "

#         else:

#             reviews=reviews.get_text().strip()



#         house_info = hotel_info(link)







#         hotelObject = {

#             "Hotel_Name" : HotelName.get_text().strip(),

#             "Address" : Address,

#             "Hotel_URL" : link,

#             "mainprice" : mainprice,

#             "tax" : tax,

#             "reviews" : reviews,

#             "house_info" : house_info

#             }  



#         conn = new_conn(link)

#         for data in conn.findAll('div', attrs={"class": "hInfo row"}):

#             for data2 in data.find_all('li'):

#                 atags = data2.find('small')

#                 btags = data2.find('span')

#                 atags = atags.get_text().strip()

#                 btags = btags.get_text().strip()

#                 hotelObject[atags] = btags





#         hotel_list.append(hotelObject) 

#     return hotel_list    

# #navigate to multiple pages



# for data in content.findAll('div', attrs={"id": "willPage"}):

#     ta = data.find_all('a')

#     for l in ta:

#         lin=l.get('href')

#         link="https://me.cleartrip.com"+lin

#         hotel_list=navigate_page(link)

# for list in hotel_list:

#     print(list,"\n")