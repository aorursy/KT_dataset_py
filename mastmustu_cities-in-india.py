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
#importing Libraries

import urllib

from urllib.parse import urlparse

from urllib.request import ProxyHandler

from urllib.request import Request

from urllib.parse import urlencode

from urllib.request import build_opener

from bs4 import BeautifulSoup

import pandas as pd
url = 'https://service.unece.org/trade/locode/in.htm'

user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
req = Request(url, headers={'User-Agent': user_agent})

opener = build_opener()

content = opener.open(req).read()

bs_x = BeautifulSoup(content,'html.parser')

bs_x
#wikitable sortable jquery-tablesorter

search_table = bs_x.findAll('tr')

print(len(search_table), type(search_table))



data = []

for table in search_table:

        data.append(table.text.splitlines())

        

print(len(data))
#print(data[4:])



data2 = data[4:]

rows = []

for x in data2:

    row= []

    print(x)

#     if x[1] == "Population":

#         break;

    row.append("INDIA")

    row.append(x[4])

    row.append(x[5])



    

    rows.append(row)



df = pd.DataFrame(rows)

df.columns =['Country','City' ,'State']

df.head()