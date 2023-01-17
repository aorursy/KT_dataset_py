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
import requests

page = requests.get("https://www.timeanddate.com/weather/india/chennai/historic")
from bs4 import BeautifulSoup

soup=BeautifulSoup(page.content,"html.parser")
table=soup.find_all("table",{"class":"zebra tb-wt fw va-m tb-hover"})

l=[]

for i,items in enumerate(table):

    for i,row in enumerate(items.find_all("tr")):

        d = {}

        try:

#             print(i , row.find_all("td",{"class":""})[0].text)

            d['Temp'] = row.find_all("td",{"class":""})[0].text

        except:

            d['Temp'] = np.nan

            

        try:

#             print(i , row.find("td",{"class":"small"}).text)

            d['Weather'] = row.find("td",{"class":"small"}).text

        except:

            d['Weather']= np.nan

            

        try:   

#             print(i , row.find_all("td",{"class":"sep"})[0].text)

            d['Wind'] = row.find_all("td",{"class":"sep"})[0].text

        except:

            d['Wind'] = np.nan

            

        try:  

#             print(i, row.find("span",{"class":"comp sa16"})['title'])

            d['Direction'] = row.find("span")["title"]

        except:

            try:

                d['Direction'] = row.find("span",{"class":"comp sa16"})["title"]

            except:

                d['Direction'] = np.nan

            

        try:

#             print(i , row.find_all("td",{"class":""})[2].text)

            d['Humidity'] = row.find_all("td",{"class":""})[2].text

        except:

            d['Humidity'] = np.nan

        

        try:

#             print(i , row.find_all("td",{"class":"sep"})[1].text)

            d['Barometer'] =  row.find_all("td",{"class":"sep"})[1].text

        except:

            d['Barometer'] = np.nan

    

        try:

#             print(i , row.find_all("td",{"class":""})[3].text)

            d['Visibility'] =  row.find_all("td",{"class":""})[3].text

        except:

             d['Visibility'] = np.nan

                

        l.append(d)



  



import pandas

df = pandas.DataFrame(l)
df2 = df.dropna(how = 'all')

df2 = df2.reset_index()

df2.pop('index')

df2['Barometer'] = df2['Barometer'].str.extract('(\d+\.\d+)') + r'"Hg'

df2['Weather'] = df2['Weather'].str.replace(" ","")

df2['Visibility'] = df2['Visibility'].str.extract('(\d+)') + 'mi'

df2['Wind'] = df2['Wind'].str.extract('(\d+)') + 'mph'

df2['Temp'] = df2['Temp'].str.extract('(\d+)') + u'\N{DEGREE SIGN}' + 'F'

df2.to_csv('data.csv', columns = ['Temp', 'Weather', 'Wind', 'Direction', 'Humidity', 'Barometer',

       'Visibility'])
df2