# from lxml import etree

# url_newslookup_daily = 'http://downloads.newslookup.com/daily.xml'

# # https://lxml.de/parsing.html

# root = etree.parse(url_newslookup_daily)

# https://www.journaldev.com/18043/python-lxml

#print(etree.tostring(root, pretty_print=True).decode("utf-8"))

# https://www.w3resource.com/python-exercises/basic/python-basic-1-exercise-8.php

import bs4

from bs4 import BeautifulSoup as soup

from urllib.request import urlopen



news_url="http://downloads.newslookup.com/daily.xml"

Client=urlopen(news_url)

xml_page=Client.read()

Client.close()



soup_page=soup(xml_page,"xml")

news_list=soup_page.findAll("item")

# Print news title, url and publish date

for news in news_list[:10]:

    print(news.title.text)

    print(news.link.text)

    if news.description:

        print(news.description.text)

    print(news.pubdate.text)

    print('Referrer : '+ news.referrer.text)

    print("-"*40)


