# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from bs4 import BeautifulSoup

import requests

from tqdm import tqdm
#省->市区

#url = 'http://www.stats.gov.cn/tjsj/tjbz/tjyqhdmhcxhfdm/2018/64.html'

url = 'http://www.stats.gov.cn/tjsj/tjbz/tjyqhdmhcxhfdm/2018/62.html'

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 LBBROWSER'}

res = requests.get(url,headers=headers)

soup = BeautifulSoup(res.content,'lxml')



city_href = []

city_num = []

city_name = []

for city in soup.find_all('tr',class_='citytr'):

    i = 0

    for city_param in city.find_all('a'):

        #print(len(city.findall('a')))

        if i % 2 == 0:

            city_num.append(city_param.get_text())

            city_href.append(url[:-5]+city_param['href'][2:])

        else:

            city_name.append(city_param.get_text())

        i += 1

city_name
City_df = pd.DataFrame({'city_name':city_name,'city_code':city_num})

City_df.to_csv('city.csv',index=None)

City_df.head()
#市->区（县）

def city_to_county(url):

    countytr_href = []

    countytr_num = []

    countytr_name = []

    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 LBBROWSER'}

    res = requests.get(url,headers=headers)

    soup = BeautifulSoup(res.content,'lxml')



    for countytr in soup.find_all('tr',class_='countytr')[1:]:

        i = 0

        for countytr_param in countytr.find_all('a'):

            if i % 2 == 0:

                #print(countytr_param['href'])

                countytr_href.append(url[:-9]+countytr_param['href'])

                countytr_num.append(countytr_param.get_text())

            else:

                countytr_name.append(countytr_param.get_text())

            i += 1

    return countytr_href,countytr_name,countytr_num





county_nums = []

county_names = []

county_urls = []

for url in tqdm(city_href):

    url,name,num = city_to_county(url)

    county_names += name

    county_nums += num

    county_urls += url
county_df = pd.DataFrame({'county_name':county_names,'county_code':county_nums})

county_df.to_csv('county.csv',index=False)

county_df.head()
#县->街道

def county_to_town(url):  

    towntr_href = []

    towntr_num = []

    towntr_name = []

    #url = countytr_href[0]

    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 LBBROWSER'}

    res = requests.get(url,headers=headers)

    soup = BeautifulSoup(res.content,'lxml')

    for towntr in soup.find_all('tr',class_='towntr'):

        i = 0

        for towntr_param in towntr.find_all('a'):

            if i % 2 == 0:

                #print(towntr_param['href'])

                towntr_href.append(url[:-11]+towntr_param['href'])

                towntr_num.append(towntr_param.get_text())

            else:

                towntr_name.append(towntr_param.get_text())

            i += 1

    return towntr_href,towntr_name,towntr_num
town_nums = []

town_names = []

town_urls = []

for url in tqdm(county_urls):

    url,name,num = county_to_town(url)

    town_names += name

    town_nums += num

    town_urls += url
town_df = pd.DataFrame({'town_name':town_names,'town_code':town_nums})

town_df.to_csv('town.csv',index=False)

town_df.head()
# #县->街道

# towntr_href = []

# towntr_num = []

# towntr_name = []

# url = countytr_href[0]

# headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 LBBROWSER'}

# res = requests.get(url,headers=headers)

# soup = BeautifulSoup(res.content,'lxml')



# #print(soup.find_all('tr',class_='towntr'))

# for towntr in soup.find_all('tr',class_='towntr'):

#     i = 0

#     for towntr_param in towntr.find_all('a'):

#         if i % 2 == 0:

#             #print(towntr_param['href'])

#             towntr_href.append(url[:-11]+towntr_param['href'])

#             towntr_num.append(towntr_param.get_text())

#         else:

#             towntr_name.append(towntr_param.get_text())

#         i += 1
#街道->居委会

def town_to_villager(url):

    villagetr_num = []

    villagetr_code = []

    villagetr_name = []

    #url = towntr_href[0]

    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 LBBROWSER'}

    res = requests.get(url,headers=headers)

    soup = BeautifulSoup(res.content,'lxml')



    #print(soup.find_all('tr',class_='towntr'))

    for villagetr in soup.find_all('tr',class_='villagetr'):

        #print(villagetr)

        i = 0

        for villagetr_param in villagetr.find_all('td'):

            if i % 3 == 0:

                villagetr_num.append(villagetr_param.get_text())

            else:

                if i % 3 == 1:

                    villagetr_code.append(villagetr_param.get_text())

                else:

                    villagetr_name.append(villagetr_param.get_text())

            i += 1

    return villagetr_code,villagetr_name,villagetr_num
import time
village_nums = []

village_names = []

village_codes = []

i = 0

for url in tqdm(town_urls):

    if i % 21 == 0:

        time.sleep(5)

    i += 1

    code,name,num = town_to_villager(url)

    village_names += name

    village_nums += num

    village_codes += code
village_df = pd.DataFrame({'village_name':village_names,'town_code':village_nums,'town_new_code':village_codes})

village_df.to_csv('town.csv',index=False)

village_df.head()
num = city_num + county_nums + town_nums + village_nums

name = city_name + county_names + town_names + village_names
code_df = pd.DataFrame({'name':name,'code':num})

code_df.to_csv('code.csv',index=False)

code_df.head()