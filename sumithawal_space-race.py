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
import seaborn as sns 

import matplotlib.pyplot as plt 

from wordcloud import WordCloud 

import re 

import string 

from matplotlib import style 

import datetime 

from  datetime import date 

style.use('fivethirtyeight')
df= pd.read_csv('../input/all-space-missions-from-1957/Space_Corrected.csv')
# we will create a dataset to make changes in the original  dataset 

df1 = df.copy()
# Getting the countrywise location of the launches

for i in range(len(df1['Location'])):

    example = df1['Location'][i].split()

    df1['Location'][i] = example[-1]
def plot_countries(df):

    df = df[~df['Location'].isin(['Sea','Canaria','Site','Facility'])]

    locations  =  df['Location'].value_counts().index

    index = [ x for x in range(len(locations))]

    y_vals = df['Location'].value_counts().values

    

    plt.figure(figsize=(12,4))

    plt.bar(index,y_vals)

    plt.xticks(index,locations,rotation=45)

    plt.xlabel('Countries',fontsize=17)

    plt.ylabel('Launches',fontsize=17)

    plt.plot()

    



plot_countries(df1)
def plot_budget(df):

    budget = df[' Rocket'].value_counts()

    plt.figure(figsize=(20,4))

    budget.plot(kind='hist')

    index= [x for x in range(len(budget))]

#     plt.xticks(index,budget.index,rotation=90)

    plt.xlabel('Budget',fontsize=17)

    plt.ylabel('Frequency of budget',fontsize=17)

    plt.show()



plot_budget(df1)
df1['Status Mission'].value_counts()

plt.pie(df1['Status Mission'].value_counts(),

       labels=df1['Status Mission'].value_counts().index,

       startangle=45,explode=[0.1,0.2,0.3,0.4])

plt.axis('equal')

plt.show()
def return_dates(df):

    df['Datum'].dropna(inplace=True)

    for i in range(len(df)):

        word_ls = df1['Datum'][i].split()

        year = word_ls[3]

        month  =  word_ls[1]

        datetime_object = datetime.datetime.strptime(month, "%b")

        month_number = datetime_object.month

        df['Datum'][i]=  '{}-{}'.format(month_number,year)

    return df

df1 =return_dates(df1)

df1['Datum'] = pd.to_datetime(df1['Datum'])

df1.Datum.dtype

def missions_over_years(df):

    df1= df.groupby(df['Datum'].dt.year).count()

    x = df1.index

    y = df1['Status Mission'].tolist()

    plt.figure(figsize=(14,8))

    plt.grid()

    plt.plot(x,y)

    plt.xlabel('Years',fontsize=20)

    plt.ylabel('Number of Missions',fontsize=20)

    plt.show()

    

missions_over_years(df1)
#  Plotting the Details of the rockets 

def word_cloud1(df):

    word_ls=[]

    for i in range(len(df)):

        word_ls += df['Detail'][i].split()

    wordcloud = WordCloud(width=350,height=250).generate(''.join(word_ls))

    plt.figure(figsize=(19,9))

    plt.axis('off')

    plt.title('Rocket Names',fontsize=20)

    plt.imshow(wordcloud)

    plt.show()

word_cloud1(df1)
# plotting the wordcloud of Company names 

def word_cloud2(df):

    word_ls=[]

    for i in range(len(df)):

        word_ls += df['Company Name'][i]

    wordcloud = WordCloud(width=250,height=200).generate(''.join(word_ls))

    plt.figure(figsize=(9,8))

    plt.axis('off')

    plt.title('Rocket Companies',fontsize=20)

    plt.imshow(wordcloud)

    plt.show()

word_cloud2(df1)