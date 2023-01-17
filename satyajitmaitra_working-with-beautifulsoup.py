# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import requests as req
URL = "https://en.wikipedia.org/wiki/Time%E2%80%93frequency_analysis"
request = req.get(URL)
request.content[:1000]
from bs4 import BeautifulSoup

soup = BeautifulSoup(request.content,"html.parser")
print(soup)
soup.prettify()
title = soup.h1
print(title)
image = soup.img
print(image)
table = soup.table
print(table)
tables = soup.find_all("table")
print(len(tables))
lists = soup.find_all("li")
print(len(lists))
#finding all links
links = soup.find_all("a")
print(len(links))
links[:3]
filter_attr = {"class":"mw-jump-link"}
links_filter = soup.find_all("a",filter_attr)
print(links_filter)
filter_attr = {"class":"noprint"}
soup.find_all("a",filter_attr)
attr_filter = {"class":"fn"}
soup.find_all(None,attr_filter)
import json


link = "https://jsonplaceholder.typicode.com/posts"
requests = req.get(link)
data = json.loads(requests.content)
data[:2]
r = req.post(link)
data = json.loads(r.content)
print(data)
input_data = {"title":"test_data",
             "user_id":5}

r = req.post(link,input_data)
data = json.loads(r.content)
print(data)
