

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import requests

import urllib.request

import time

from bs4 import BeautifulSoup

import os



url = "https://www.nasscom.in/members-listing"

base_url = "https://www.nasscom.in/members-listing?title=&page="
response = requests.get(url)
soup = BeautifulSoup(response.text)

# list_of_attachments_tags = soup.find("div",{"class":"attachment attachment-before"}).find("div",{"class":"view-content"}).findAll("span",{"class":"views-summary views-summary-unformatted"})

# list_of_attachments = [base_url+e.find('a').get("href") for e in list_of_attachments_tags]

# print(list_of_attachments)

#data=soup.find("div", {"class": "item-list"})



def getLastPageNumber(soup):

    last_page = soup.find('li',{"class":'pager-last last'}).find('a').get('href').split("&")[-1].split("=")[-1]

    return int(last_page)



getLastPageNumber(soup)

def getCompanyNamesForPage(data):

    elements = data.findAll("div",{"class":"views-field views-field-title"})

    names = []

    for e in elements:

        names.append(e.find("span",{"class": "field-content"}).contents[0])

    return names



def getCityNamesForPage(data):

    elements =  data.findAll("div",{"class":"views-field views-field-field-city-members-list"})

    cities = []

    for e in elements:

        cities.append(e.find("div",{"class": "field-content"}).contents[0])

    return cities





def getWebLinksForPage(data):

    elements =  data.findAll("div",{"class":"views-field views-field-field-website-member"})

    links = []

    for e in elements:

        links.append(e.find("div",{"class": "field-content"}).find('a').get("href"))

    return links

# print(getCompanyNamesForPage(data))

# print(getCityNamesForPage(data))

# print(getWebLinksForPage(data))
def getPageData(souppageobj):

    return souppageobj.find("div", {"class": "item-list"})
names,cities, weblinks = [],[],[]

for i in range(getLastPageNumber(soup)+1):

    if i == 0:

        url = "https://www.nasscom.in/members-listing"

        response = requests.get(url)

        soup_pageobj =  BeautifulSoup(response.text)

    else:

        url = base_url+str(i)

    

    response = requests.get(url)

    soup_pageobj =  BeautifulSoup(response.text)

    data= getPageData(soup_pageobj)

#     print(url,data)

    names += getCompanyNamesForPage(data)

    cities += getCityNamesForPage(data)

#     weblinks += getWebLinksForPage(data)

    

    print("Done with page:",i)

    

# print(names)

        
dataframe = pd.DataFrame({"Company Name": names,"City": cities})

print("Sample Data: \n", dataframe.head(),"\n\n\n")

print("Total Companies: ",len(dataframe))
dataframe.to_csv("Company_Data.csv")
dataframe.to_excel("Company_Data.xls")