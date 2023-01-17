import requests

from bs4 import BeautifulSoup

import urllib

from urllib.request import urlopen
html = urlopen('https://www.nhsinform.scot/illnesses-and-conditions/a-to-z')

bs = BeautifulSoup(html, "html.parser")
headers = bs.find_all('h2', {'class':'module__title'})

# print('List all the h2 tags :', *headers, sep='\n\n')
titles = list(map(lambda h: h.text.strip(), headers))

titles