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
#using Beautifulsoup,requests and urllib

from bs4 import BeautifulSoup

from urllib.parse import urlsplit

from urllib.parse import urlparse

from collections import deque

from bs4 import SoupStrainer

import requests

from urllib.request import Request

from urllib.request import build_opener

from urllib.request import ProxyHandler

from urllib.parse import urlencode
website_url = requests.get('http://www.unece.org/cefact/locode/service/location.html').text

user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
from bs4 import BeautifulSoup

soup = BeautifulSoup(website_url,'lxml')

print(soup.prettify()) #enable us to view how the tags are nested in the document
My_table = soup.find('table',{'class':'contenttable'})

My_table
#links= My_table.findAll('a')

#type(links)

import re

pattern = re.compile(r"http://service.unece.org/trade/locode/*?.htm")

links = [a['href'] for a in soup.find_all('a', href=pattern)]

links
link_list = [a['href'] for a in soup.find_all('a', href=True)]

link_list
type(link_list)
import re 

  

def Filter(string, substr): 

    return [str for str in string if

             any(sub in str for sub in substr)] 

      

# Driver code 

substr = ['https://service.unece.org/trade/locode/'] 

urls= []

urls= Filter(link_list, substr) 

urls
for url in urls: 

    req = Request(url, headers={'User-Agent': user_agent})

    opener = build_opener()

    content = opener.open(req).read()

    bs_x = BeautifulSoup(content,'html.parser')