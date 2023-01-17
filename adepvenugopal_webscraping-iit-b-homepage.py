from urllib.request import urlopen

from bs4 import BeautifulSoup

import re
html = urlopen('http://www.iitb.ac.in')

#html.read()
bs = BeautifulSoup(html.read(), 'html.parser')

#print(bs)
print(bs.h1)

print(bs.h2)

print(bs.h3)

print(bs.h4)

print(bs.h5)
nameList = bs.findAll('div', {'class': 'views-field views-field-title'})

for name in nameList:

    print(name.get_text())
nameList = bs.findAll('div', {'class': 'views-field views-field-body'})

for name in nameList:

    print(name.get_text())
nameList = bs.findAll('div', {'class': 'work-item-content'})

for name in nameList:

    print(name.get_text())
nameList = bs.findAll('li', {'class': 'leaf'})

for name in nameList:

    print(name.get_text())
nameList = bs.findAll('ul', {'class': 'menu'})

for name in nameList:

    print(name.get_text())
for link in bs.findAll('a', attrs={'href': re.compile("^http://")}):

    print(link.get('href'))