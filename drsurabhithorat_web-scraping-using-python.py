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

import requests 
from bs4 import BeautifulSoup 
import csv 
   
URL = "http://www.values.com/inspirational-quotes"
r = requests.get(URL) 
   
soup = BeautifulSoup(r.content, 'html5lib') 
   
quotes=[]  # a list to store quotes 
   
table = soup.find('div', attrs = {'id':'all_quotes'})  
   
for row in table.findAll('div', 
                         attrs = {'class':'col-6 col-lg-3 text-center margin-30px-bottom sm-margin-30px-top'}): 
    quote = {} 
    quote['theme'] = row.h5.text 
    quote['url'] = row.a['href'] 
    quote['img'] = row.img['src'] 
    quote['lines'] = row.img['alt'].split(" #")[0] 
    quote['author'] = row.img['alt'].split(" #")[1] 
    quotes.append(quote) 
    
#Result is stored in CSV file.

filename = 'inspirational_quotes.csv'
with open(filename, 'w', newline='') as f: 
    w = csv.DictWriter(f,['theme','url','img','lines','author']) 
    w.writeheader() 
    for quote in quotes: 
        w.writerow(quote)
#Check the result in output folder