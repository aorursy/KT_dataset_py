from urllib.request import urlopen

from bs4 import BeautifulSoup 



html = urlopen('http://en.wikipedia.org/wiki/Kevin_Bacon')

bs = BeautifulSoup(html, 'html.parser')

for link in bs.find_all('a'):

    if 'href' in link.attrs:

        print(link.attrs['href'])
from urllib.request import urlopen 

from bs4 import BeautifulSoup 

import re



html = urlopen('http://en.wikipedia.org/wiki/Kevin_Bacon')

bs = BeautifulSoup(html, 'html.parser')

for link in bs.find('div', {'id':'bodyContent'}).find_all(

    'a', href=re.compile('^(/wiki/)((?!:).)*$')):

    if 'href' in link.attrs:

        print(link.attrs['href'])
from urllib.request import urlopen

from bs4 import BeautifulSoup

import datetime

import random

import re



random.seed(datetime.datetime.now())

def getLinks(articleUrl):

    html = urlopen('http://en.wikipedia.org{}'.format(articleUrl))

    bs = BeautifulSoup(html, 'html.parser')

    return bs.find('div', {'id':'bodyContent'}).find_all('a', href=re.compile('^(/wiki/)((?!:).)*$'))



links = getLinks('/wiki/Kevin_Bacon')

while len(links) > 0:

    newArticle = links[random.randint(0, len(links)-1)].attrs['href']

    print(newArticle)

    links = getLinks(newArticle)
from urllib.request import urlopen

from bs4 import BeautifulSoup

import re



pages = set()

def getLinks(pageUrl):

    global pages

    html = urlopen('http://en.wikipedia.org{}'.format(pageUrl))

    bs = BeautifulSoup(html, 'html.parser')

    for link in bs.find_all('a', href=re.compile('^(/wiki/)')):

        if 'href' in link.attrs:

            if link.attrs['href'] not in pages:

                #We have encountered a new page

                newPage = link.attrs['href']

                print(newPage)

                pages.add(newPage)

                getLinks(newPage)

getLinks('')
from urllib.request import urlopen

from bs4 import BeautifulSoup

import re



pages = set()

def getLinks(pageUrl):

    global pages

    html = urlopen('http://en.wikipedia.org{}'.format(pageUrl))

    bs = BeautifulSoup(html, 'html.parser')

    try:

        print(bs.h1.get_text())

        print(bs.find(id ='mw-content-text').find_all('p')[0])

        print(bs.find(id='ca-edit').find('span').find('a').attrs['href'])

    except AttributeError:

        print('This page is missing something! Continuing.')

    

    for link in bs.find_all('a', href=re.compile('^(/wiki/)')):

        if 'href' in link.attrs:

            if link.attrs['href'] not in pages:

                #We have encountered a new page

                newPage = link.attrs['href']

                print('-'*20)

                print(newPage)

                pages.add(newPage)

                getLinks(newPage)

getLinks('') 
from urllib.request import urlopen

from urllib.parse import urlparse

from bs4 import BeautifulSoup

import re

import datetime

import random



pages = set()

random.seed(datetime.datetime.now())



#Retrieves a list of all Internal links found on a page

def getInternalLinks(bs, includeUrl):

    includeUrl = '{}://{}'.format(urlparse(includeUrl).scheme, urlparse(includeUrl).netloc)

    internalLinks = []

    #Finds all links that begin with a "/"

    for link in bs.find_all('a', href=re.compile('^(/|.*'+includeUrl+')')):

        if link.attrs['href'] is not None:

            if link.attrs['href'] not in internalLinks:

                if(link.attrs['href'].startswith('/')):

                    internalLinks.append(includeUrl+link.attrs['href'])

                else:

                    internalLinks.append(link.attrs['href'])

    return internalLinks

            

#Retrieves a list of all external links found on a page

def getExternalLinks(bs, excludeUrl):

    externalLinks = []

    #Finds all links that start with "http" that do

    #not contain the current URL

    for link in bs.find_all('a', href=re.compile('^(http|www)((?!'+excludeUrl+').)*$')):

        if link.attrs['href'] is not None:

            if link.attrs['href'] not in externalLinks:

                externalLinks.append(link.attrs['href'])

    return externalLinks



def getRandomExternalLink(startingPage):

    html = urlopen(startingPage)

    bs = BeautifulSoup(html, 'html.parser')

    externalLinks = getExternalLinks(bs, urlparse(startingPage).netloc)

    if len(externalLinks) == 0:

        print('No external links, looking around the site for one')

        domain = '{}://{}'.format(urlparse(startingPage).scheme, urlparse(startingPage).netloc)

        internalLinks = getInternalLinks(bs, domain)

        return getRandomExternalLink(internalLinks[random.randint(0,

                                    len(internalLinks)-1)])

    else:

        return externalLinks[random.randint(0, len(externalLinks)-1)]

    

def followExternalOnly(startingSite):

    externalLink = getRandomExternalLink(startingSite)

    print('Random external link is: {}'.format(externalLink))

    followExternalOnly(externalLink)

            

followExternalOnly('http://oreilly.com')
# Collects a list of all external URLs found on the site

allExtLinks = set()

allIntLinks = set()





def getAllExternalLinks(siteUrl):

    html = urlopen(siteUrl)

    domain = '{}://{}'.format(urlparse(siteUrl).scheme,

                              urlparse(siteUrl).netloc)

    bs = BeautifulSoup(html, 'html.parser')

    internalLinks = getInternalLinks(bs, domain)

    externalLinks = getExternalLinks(bs, domain)



    for link in externalLinks:

        if link not in allExtLinks:

            allExtLinks.add(link)

            print(link)

    for link in internalLinks:

        if link not in allIntLinks:

            allIntLinks.add(link)

            getAllExternalLinks(link)





allIntLinks.add('http://oreilly.com')

getAllExternalLinks('http://oreilly.com')