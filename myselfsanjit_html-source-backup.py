#for single url

import pandas as pd

import numpy as np

from bs4 import BeautifulSoup

import requests 

import urllib







url=input('Kindly enter a url: ')





response = requests.get(url)

html_soup = BeautifulSoup(response.text)

main_content = html_soup.find('body')

# main_content = html_soup.find("div", {"class":"product-description"})







filename = url.rsplit('/', 1)[1]

file = open(filename, "w")

file.write(str(main_content))

file.close()

print ("Backup is Created")
#for multiple url separated by comma



from bs4 import BeautifulSoup

import requests 

import urllib



url_in=input('Kindly inter a url or list of urls sepatared by commams:  ').split(sep=',')





for x in url_in:

    response = requests.get(x)

    html_soup = BeautifulSoup(response.text)

    main_content = html_soup.find('body')

    # main_content = html_soup.find("div", {"class":"product-description"})

    filename = x.rsplit('/', 1)[1]

    file = open(filename, "w")

    file.write(str(main_content))

    file.close()

print ("Backup is Created")

    

    