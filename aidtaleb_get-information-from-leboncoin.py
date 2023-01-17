import urllib.request
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import numpy as np # library to handle data in a vectorized manner
import requests
import pandas as pd # library for data analsysis
!pip install requests_cache
import requests_cache
import json # library to handle JSON files
!pip install shadow_useragent
import shadow_useragent
import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
# import k-means from clustering stage
from sklearn.cluster import KMeans
!pip install folium
import folium # map rendering library
requests_cache.install_cache("bases_scraping", expire_after=10e5)
print('Libraries imported.')


requests_cache.install_cache("bases_scraping", expire_after=10e5)
print('Libraries imported.')
url = 'https://www.leboncoin.fr/recherche/?category=2&locations=rn_12&page='
headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36',
'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
'Accept-Language': 'en-US,en;q=0.5',
'Accept-Encoding': 'gzip, deflate',
'Connection': 'keep-alive',
'Upgrade-Insecure-Requests': '1'}
s = requests.Session()
s.headers.update(headers)
s.get('https://www.leboncoin.fr/')

r = s.get(url)

def get_pages(token, nb):
    pages = []
    for i in range(1,nb+1):
        j = token + str(i)
        pages.append(j)
    return pages
pages = get_pages(url,30)

response=[]
for i in pages:
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36',
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
               'Accept-Language': 'en-US,en;q=0.5',
               'Accept-Encoding': 'gzip, deflate',
               'Connection': 'keep-alive',
               'Upgrade-Insecure-Requests': '1'}
    s = requests.Session()
    s.headers.update(headers)
    s.get('https://www.leboncoin.fr/')
    response.append(s.get(i))

soup=[]
for i in range(len(response)):
    soup.append(BeautifulSoup(response[i].text, 'html.parser'))

print(soup[0])

data1=[]
for k in range(len(soup)):
    data1.append(soup[k].findAll("script"))

data2 =[]

for j in range(len(data1)):
    if len(data1[j])==15 : 
        data2.append(data1[j][13])

X=[]
for p in range(len(data2)):
    X.append(str(data2[p]))



data1[0]
data3=[]
data4=[]
for q in range(len(X)):
    data3.append(X[q].split('status'))
    for i in range(len(data3[q])):
        data4.append(data3[q][i].split(','))
        


data4[1]

data_subject=[]
data_body=[]
data_url=[]
data_price=[]
data_city=[]
for k in  range(len(data4)):
    for item in data4[k]: 
        if "subject" in item: 
            data_subject.append(item)
        elif "body" in item : 
            data_body.append(item)
        elif "htm" in item : 
            data_url.append(item)
        elif "price" in item : 
            if "calendar" not in item :
                data_price.append(item)
        elif 'city_label'in item:
            data_city.append(item)


data_body[2]
subject=[]
final_sub=[]
body=[]
final_body=[]
price=[]
final_price=[]
city=[]
final_city=[]
url=[]
final_url=[]
mark=[]
index=min(len(data_subject),len(data_price),len(data_city),len(data_url),len(data_body))
for i in range(index): 
    subject.append(data_subject[i].split(':'))
    final_sub.append(subject[i][1])
    if "audi" in final_sub[i].lower():
        mark.append("audi")
    elif "bmw" in final_sub[i].lower():
        mark.append("bmw")
    elif  "mercedes"in final_sub[i].lower():
        mark.append("Mercedes")
    elif ("mini"or"appel") in final_sub[i].lower():
        mark.append("Mini")
    elif "opel" in  final_sub[i].lower():
        mark.append("Opel")
    elif  "volkswagen" in  final_sub[i].lower():
        mark.append("Volkswagen")
    elif "alpina" in  final_sub[i].lower():
        mark.append("Alpina")
    elif "smart"in  final_sub[i].lower():
        mark.append("Smart")
    else: 
        mark.append("other mark")
        
    body.append(data_body[i].split(':"'))
    if len(body[i])==2:
        final_body.append(body[i][1])
    price.append(data_price[i].split(':'))
    final_price.append(price[i][1])
    city.append(data_city[i].split(':'))
    final_city.append(city[i][1])
    url.append(data_url[i].split('"'))
    final_url.append(url[i][3])

final_body
final_sub

Y=[]
X=[]
x='//'
y='/'
for l in range(len(final_url)):
    Y.append(final_url[l].split('\\u002F'))
    if len (Y[l])== 5 :
        X.append(Y[l][0]+ x +Y[l][2]+ y  + Y[l][3]+ y +Y[l][4])
    else : 
        X.append('no link to provide')


import re
numbers=[]
num=[]
prix=[]
for k in range(len(final_price)):
    numbers.append(re.findall(r'\d+',final_price[k]))
    num.append(int(numbers[k][0]))
    item=num[k]
    if item < 10 : 
        prix.append(item*100)
    elif item >= 10 and item <100 : 
         prix.append(item*10)
    else :
        prix.append(item)

prix
pc_data=pd.DataFrame()

mark
index = min(len(final_sub),len(final_body),len(prix),len(final_city),len(X))
index
X[0:index]
index = min(len(final_sub),len(final_body),len(num),len(final_city),len(X), len(mark))
ind=index-1
pc_data['PC type']=final_sub[0:ind]
pc_data["Marque"]=mark[0:ind]
pc_data['PC annonce']=final_body[0:ind]
pc_data['PC price']=prix[0:ind]
pc_data['PC city']=final_city[0:ind]
pc_data['PC link']=X[0:ind]
pc_data.head()
