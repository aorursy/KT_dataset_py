#Imports

import os

import requests

from bs4 import BeautifulSoup

import urllib.request

from IPython.display import Image
##Scrapping static pages



#Static website

url = "https://en.wikipedia.org/wiki/Quokka"

try:

    webpage = requests.get(url)

except requests.exceptions.RequestsExceptions as e:

    print(e)



#Show content of webpage

#Parse downloaded content into a BeautifulSoup object

soup = BeautifulSoup(webpage.content, "lxml")

#print(soup.prettify())
#List containing all url src from html with img tag

images_url = []

#Filter out the url with Quokka keyword

quokka_img_url = []

for link in soup.find_all('img'):

    images_url.append(link.get('src'))



#Completing the url

for img in images_url:

    if "Quokka" in img:

        quokka_img_url.append("https:" + img)

    if "Qokka" in img:

        quokka_img_url.append("https:" + img)

        

quokka_img_url
#Displaying image from url

for url in quokka_img_url:

    display(Image(url,width=300, height=300))