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
#importing libraries and building a soup object

from bs4 import BeautifulSoup

import pandas as pd

import requests

import numpy as np



url = 'https://www.worldometers.info/coronavirus/'



r = requests.get(url)

html_doc = r.text

soup = BeautifulSoup(html_doc, features='html.parser')



#print(soup.prettify())

print(soup.title.text)

datadiv=soup.find(class_='preview-main-content-form')

print(datadiv)
#extracting data from the table

covid_data =[]

row=0

for tr in soup.findAll("tr"):

    elements=[]

    column=0

    for td in tr.findAll("td"):

        if(td.text!=''):

            elements.append(td.text)

            column+=1

            #print('column: ', column)   



    covid_data.append(elements)        

    #print('row: ', row)        

    row+=1
#lets see how the data looks and we will see the data is quite complex in form of list of lists

covid_data
#now, lets clean the data and convert it into a dataframe

covid_df = pd.DataFrame(data=covid_data)

print(covid_df)

covid_df=covid_df.drop(12,axis=1)

covid_df.columns
#It is visible there are a lot of unwanted data in rows and columns, so lets get rid of that

covid_df = covid_df[9:100]

covid_df=covid_df.reset_index()

covid_df = covid_df.drop('index',axis=1)
#cleaneing dataframe and having data of around 100 countries

covid_df = covid_df[[1,2,3,4,5,6]]

covid_df = covid_df.rename(columns={1:'Country',2:'Total Cases',3:'Cases on 14thOct',4:'Total Deaths',5:'Deaths on 14thOct',6:'Total Recovered'})

covid_df
#removing the commas from dataframe it's somehow causing the error in plotting

import numpy as np

covid_df = covid_df.replace(",","",regex=True)

covid_df["Total Cases"] = covid_df["Total Cases"].astype(float)
#importing plotting libraries and showing a bar plot of total cases as per 14thOct

import matplotlib.pyplot as plt

%matplotlib inline

covid_df.plot.barh(x="Country",y=["Total Cases"],figsize=(30,25),fontsize=12,rot=0,xticks=covid_df.index)

plt.title('Total Cases as on 14thOct')

plt.xlabel("Country",fontsize=12)

plt.ylabel("Total Cases",fontsize=12)
#saving the dataframe

covid_df.to_csv('Covid-19.csv',sep=',')