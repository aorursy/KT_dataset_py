import urllib.request

import urllib.parse

import re



import bs4

from bs4 import BeautifulSoup

import requests



query_string = urllib.parse.urlencode({"search_query" : input()})

html_content = urllib.request.urlopen("http://www.youtube.com/results?" + query_string)

search_results = re.findall(r'href=\"\/watch\?v=(.{11})', html_content.read().decode())



result = []

for i in range(5):

    link = "http://www.youtube.com/watch?v=" + search_results[i]

    r = requests.get(link).text

    soup = BeautifulSoup(r, "lxml")

    title = soup.find("title").text

    result.append((title, link))

result