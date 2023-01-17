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
import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns 
Df = pd.read_csv("/kaggle/input/indian-startup-funding/startup_funding.csv", encoding='utf-8')
Df.head()
Df.tail()
Names_to_be_changed = {'Investorsxe2x80x99 Name':'Investor Name','InvestmentnType': 'Investment Type'}
Df.rename(columns=Names_to_be_changed,inplace=True)
invalid_index = []

for i in range(len(Df['Date ddmmyyyy'])):

    try:

        #print(i)

        pd.to_datetime(Df['Date ddmmyyyy'][i],format='%d/%m/%Y')

    except:

        print(i)

        invalid_index.append(i)
Df['Date ddmmyyyy'][invalid_index]
Df['Date ddmmyyyy'][invalid_index]

Df['Date ddmmyyyy'][2570]

Correct_dates = ['05/07/2018','01/07/2015','10/07/2015','12/05/2015','12/05/2015','13/04/2015','15/01/2015','22/01/2015']

Df['Date ddmmyyyy'].iloc[invalid_index] = Correct_dates

Df['Date ddmmyyyy'] = pd.to_datetime(Df['Date ddmmyyyy'],format='%d/%m/%Y')

Df['Year'] = [x.year for x in Df['Date ddmmyyyy']]

Df['Month'] = [x.month for x in Df['Date ddmmyyyy']]

Df['Day']= [x.day_name() for x in Df['Date ddmmyyyy']]

Df['Quarter'] = [x.quarter for x in Df['Date ddmmyyyy']]
Df['Amount in USD'] = Df['Amount in USD'].apply(lambda x: str(x).replace(',','').replace('\\\\xc2\\\\xa0','').replace('+',''))

Df['Amount in USD'] = Df['Amount in USD'].apply(lambda x: x.replace('nan','undisclosed').replace('N/A','undisclosed')).replace('Undisclosed','undisclosed')
Df['undisclosed'] = 0

Df.loc[Df['Amount in USD'] == 'undisclosed',['undisclosed']] = 1

Df.loc[Df['Amount in USD'] == 'undisclosed',['Amount in USD']] = 0

Df['Amount in USD'] = pd.to_numeric(Df['Amount in USD'])
plt.figure()

Df.groupby('Year')['Amount in USD'].agg(['count']).plot(kind = 'bar',figsize = (12,8))

plt.xlabel('Year')

plt.ylabel('Number of deals')
plt.figure()

Df.groupby('Year')['Amount in USD'].agg(['sum']).plot(kind = 'bar',figsize = (12,8))

plt.xlabel('Year')

plt.ylabel('Total Funding')
fig = plt.figure(figsize = (12,8))

ax = fig.add_subplot(1,1,1)

Funding_by_month = Df.groupby(['Year','Quarter'])['Amount in USD'].agg(['sum']).reset_index()

sns.barplot(x = 'Quarter',y = 'sum',hue = 'Year',data = Funding_by_month,linewidth=2.5)

ax.set_xlabel("Quarter")

ax.set_ylabel("Sum")
import re

def clean_names(some_text):

    some_text = str(some_text)

    some_text = some_text.replace('\\\\n',' ').replace('\\\\xe2\\\\x80\\\\x99',' ').replace('\\\\xc3\\\\x98',' ').replace('\\\\n\\\\n','').replace("\\\\xc2\\\\xa0",'')

    return some_text
Df['Industry Vertical'] = Df['Industry Vertical'].apply(lambda x : clean_names(x))
Df['Industry Vertical'].replace(['eCommerce','ECommerce','Ecommerce','E-commerce'],'E-Commerce',inplace = True)
Df['Industry Vertical'].value_counts().drop(labels = 'nan').sort_values(ascending=False)[0:10].plot(kind='bar',figsize = (12,8))


for name,group in Df.groupby('Year'): 

    print(name)

    try:

        plt.figure(figsize = (12,8))

        group['Industry Vertical'].value_counts().drop(labels = 'nan').sort_values(ascending=False)[0:10].plot(kind='bar')

        plt.title("Top 10 Industry")

        plt.xlabel("Startup funded in Year {}".format(name))

    except:

        plt.figure(figsize = (12,8))

        group['Industry Vertical'].value_counts().sort_values(ascending=False)[0:10].plot(kind='bar')

        plt.title("Top 10 Industry")

        plt.xlabel("Startup funded in Year {}".format(name))
Df['Investor Name'] = Df['Investor Name'].apply(lambda x: clean_names(x))
def get_individual_names(some_string):

    some_string = some_string.split('and')

    some_string = ','.join(some_string)

    some_string = some_string.split('&')

    some_string = ','.join(some_string)

    some_string = some_string.split(',')

    modified_string_list = []

    for string in some_string:

        string = string.strip()

        modified_string_list.append(string)

    return modified_string_list
Investor_names = Df['Investor Name'].apply(lambda x: get_individual_names(x))
names_to_be_removed = ['',' ','an','An','S','Undisclosed Investors','Undisclosed investors','undisclosed investors','Undisclosed','undisclosed','others','Others','nan']
Investor_names_list = []

for investors in Investor_names:

    for investor in investors:

        if(investor not in names_to_be_removed):

            Investor_names_list.append(investor)

   
from collections import Counter

Investors = pd.Series(Counter(Investor_names_list))

Investors.sort_values(ascending = False)[0:10].plot(kind = 'bar',figsize = (12,8))
Df['Startup Name'].value_counts().sort_values(ascending=False)[0:10].plot(kind='bar',figsize= (12,8))
Df['City  Location'] = Df['City  Location'].apply(lambda x: clean_names(x))
Df['City  Location'].replace('Gurgaon','Gurugram',inplace = True)

Df['City  Location'].replace('Bengaluru','Bangalore',inplace = True)
Df['City  Location'].value_counts().sort_values(ascending=False).drop('nan')[0:10].plot(kind='bar',figsize = (12,8))
Df['Investment Type'] = Df['Investment Type'].apply(lambda x : clean_names(x))
Df['Investment Type'].value_counts().sort_values(ascending = False)[0:10].plot(kind = 'bar',figsize = (12,8))
fig = plt.figure(figsize = (12,8))

Funding_by_type = Df.groupby(['Investment Type'])['Amount in USD'].agg(['sum']).sort_values(by = 'sum',ascending = False).reset_index()[0:10]

sns.barplot(x = 'Investment Type',y = 'sum',data = Funding_by_type,linewidth=2.5)

plt.xticks(rotation = 90)

plt.ylabel('Total investment')