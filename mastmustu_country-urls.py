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

#from urllib2.request import build_opener as bo
#website_url = requests.get('http://www.unece.org/cefact/locode/service/location.html').text



url = 'http://www.unece.org/cefact/locode/service/location.html'

user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'

req = Request(url, headers={'User-Agent': user_agent})

opener = build_opener()

content = opener.open(req).read()

bs_x = BeautifulSoup(content,'html.parser')
#bs_x
My_links = bs_x.findAll('a')

My_links
len(My_links)
start ='Afghanistan'

end = 'Zimbabwe'

list_of_urls = []



link_list = [a['href'] for a in bs_x.find_all('a', href=True)]





import re 

  

def Filter(string, substr): 

    return [str for str in string if

             any(sub in str for sub in substr)] 

      

# Driver code 

substr = ['https://service.unece.org/trade/locode/'] 

urls= []

urls= Filter(link_list, substr) 

urls




search_table = bs_x.findAll('tr')

print(len(search_table), type(search_table))



data = []

for table in search_table:

        data.append(table.text.splitlines())

        

print(len(data))
data2  = data[1:]



rows = []

for x in data2:

    row= []



    

    row.append(x[0][:2])

    rows.append(row)



df = pd.DataFrame(rows)

df.columns =['Country code']

df.head()
df.tail()
# special handling for USA



US_URLS = [ x for x in urls if x.startswith('https://service.unece.org/trade/locode/us')]
for x in US_URLS :

    urls.remove(x)

    

len(urls)
df = df[df['Country code'] != 'US']



df.shape
df['URLS'] = urls
df.tail()
df.loc[len(df) +1] = ['US' ,US_URLS]

df.tail()



#df.loc[len(df)]=['8/19/2014','Jun','Fly','98765']
df.tail()
# scarpe all countries 

# https://www.geonames.org/countries/



url = 'https://www.iban.com/country-codes'

user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'





req = Request(url, headers={'User-Agent': user_agent})

opener = build_opener()

content = opener.open(req).read()

countries = BeautifulSoup(content,'html.parser')



My_links = countries.findAll('tr')

My_links






data = []

for table in My_links:

        data.append(table.text.splitlines())

        

print(len(data))
data
data2  = data[1:]



rows = []

for x in data2:

    row= []



    row.append(x[1])

    row.append(x[2])

    row.append(x[3])

    row.append(x[4])

    rows.append(row)



country_df = pd.DataFrame(rows)

country_df.columns =['Country Name' , 'Country code' , 'Country 3 code', 'Country Numeric Code' ]

country_df.head()
final_df = country_df.merge(df , how='outer' , on = 'Country code')

final_df
country_df.shape
df.shape
final_df['URLS'].isna().sum()
final_df.to_csv('Country_URLS.csv' , index= False)
final_df.tail()
len(urls)
import time



iter_df  = final_df.dropna(axis = 0)

print(iter_df.shape)

rows = []

country_processed =''
global_iter_df = iter_df.copy()


#while (True) :

def extraction(loop_df):

    

    try :

        for index,row in loop_df.iterrows():



            time.sleep(2)

            urls = row['URLS']

            name = row['Country Name']

            code = row['Country code']

            code_3 = row['Country 3 code']

            code_no = row['Country Numeric Code']

            if isinstance(urls , list):

                for x in urls[1:] :

                    print(x)

                    req = Request(x, headers={'User-Agent': user_agent})

                    opener = build_opener()

                    content = opener.open(req).read()

                    cities = BeautifulSoup(content,'html.parser')



                    search_table = cities.findAll('tr')





                    data = []

                    for table in search_table:

                            data.append(table.text.splitlines())



                    #print(len(data))





                    data2 = data[4:]

                    print(name)

                    for x in data2:

                        row= []

                        #print(x)



                        row.append(name)

                        row.append(code)

                        row.append(code_3)

                        row.append(code_no)

                        row.append(x[3])

                        row.append(x[5])





                        rows.append(row)

                    country_processed =name

            else :

                print(name)

                req = Request(urls, headers={'User-Agent': user_agent})

                opener = build_opener()

                content = opener.open(req).read()

                cities = BeautifulSoup(content,'html.parser')



                search_table = cities.findAll('tr')





                data = []

                for table in search_table:

                        data.append(table.text.splitlines())



                #print(len(data))





                data2 = data[4:]



                for x in data2:

                    row= []

                    #print(x)



                    row.append(name)

                    row.append(code)

                    row.append(code_3)

                    row.append(code_no)

                    row.append(x[3])

                    row.append(x[5])





                    rows.append(row)

                country_processed =name

                if country_processed =='Zimbabwe' :

                    break



    except :

        print('country_processed  -->', country_processed)

        print(  "Row Number --> " ,loop_df.loc[loop_df['Country Name'] == country_processed].index[0] )

        print(loop_df.shape)

        print(loop_df.index)

        row_no = global_iter_df.loc[global_iter_df['Country Name'] == country_processed].index[0]

        new_df = global_iter_df[row_no :]

        print(new_df.shape)

        print(new_df.head())

        extraction(new_df)

    #print("out of while loop")

print("process stopped")
#print(  "Row Number --> " ,iter_df.loc[iter_df['Country Name'] == country_processed].index )

extraction(iter_df)
len(rows)
city_df = pd.DataFrame(rows)

city_df.columns =['Country Name','ISO 2' ,'ISO 3' ,'ISO Num' ,'City' ,'State']

city_df.shape

city_df.to_csv('city_data.csv', index = False)
city_df ['Country Full name'] = city_df['Country Name']

city_df['Country Name'] = city_df['Country Name'].apply(lambda x : x.split('(')[0])
#city_df

city_df['Country Name']= city_df['Country Name'].apply(lambda x : x.strip())

city_df.to_csv('final_city_wise_data.csv', index = False ,encoding='utf-8-sig')