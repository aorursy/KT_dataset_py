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
import re
import requests
import bs4
from bs4 import BeautifulSoup
url = 'https://cashkaro.com/thegoodlookbook/best-soap-brands-in-india/'
data = requests.get(url)
data
soup = BeautifulSoup(data.text, 'html.parser')
print(soup.prettify())
brnd= soup.find_all('u')
brnd
brand =[]
for i in brnd:
  a = re.findall(r'([A-Za-z]\w*)',i.text)
  brand.append(' '.join(a))
print(len(brand))
brand = brand[1:]
len(brand)

pric = soup.find_all('p')

p =[]
for i in pric:
  a = re.findall(r'([Rs.]\d+)',i.text)
  p.append(''.join(a))
    
len(p)
p
res = [] 
for i in p: 
    if i not in res: 
        res.append(i) 
del res[0]
print(len(brand))
print(len(res))
res
df = pd.DataFrame()
df['Brand'] = brand
df['Price'] = res
df['Price'] = df['Price'].str.replace('.' ,'')
df
import matplotlib.pyplot as plt 
import seaborn as sns
plt.figure(figsize=(10,7))
sns.barplot(y=df['Brand'] ,x= df['Price'])

