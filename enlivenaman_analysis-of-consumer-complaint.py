# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

pd.plotting.register_matplotlib_converters()

from datetime import datetime

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import matplotlib.dates as mdates

# This is used for fast string concatination

from io import StringIO

# Use nltk for valid words

import nltk

# Need to make hash 'dictionaries' from nltk for fast processing

import collections as co

from wordcloud import WordCloud



import warnings # current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
comcast_data = pd.read_csv("/kaggle/input/comcastcomplaints/comcast_fcc_complaints_2015.csv",parse_dates=True)

comcast_data.Date = pd.to_datetime(comcast_data.Date)
# Only interested in data with consumer complaints

customer_compaints=comcast_data[comcast_data['Customer Complaint'].notnull()]

s=StringIO()

customer_compaints['Customer Complaint'].apply(lambda x: s.write(x))



k=s.getvalue() # Array of words, with stop words removed

s.close() # Concatinated string of all comments

k=k.lower()

k=k.split()

words = co.Counter(nltk.corpus.words.words())

# removing most commonly used words from strings

stopWords =co.Counter( nltk.corpus.stopwords.words() )

k=[i for i in k if i in words and i not in stopWords]

s=" ".join(k)

c = co.Counter(k) # Collection of words
# Top 10 most common words

c.most_common(10)
# Read the whole text.

text = s

# Generate a word cloud image

wordcloud = WordCloud().generate(text)

wordcloud = WordCloud(background_color="black",max_words=len(k),max_font_size=40, relative_scaling=.8).generate(text)

plt.figure(figsize=(14,6))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
comcast_daily_complains_data = pd.DataFrame({'count':comcast_data.groupby('Date')['Ticket #'].count()})

comcast_daily_complains_data.sort_values(by='Date', ascending = True)

plt.figure(figsize=(24,10))

plt.xticks(rotation=90)

plt.title("Daily Comcast Telecom Consumer Complaints")

plt.xlabel("Date of complaint")

plt.ylabel("Number of complaints")

plt.plot(comcast_daily_complains_data, marker='o')

index=0

for i in comcast_daily_complains_data['count'].values.tolist():

    plt.annotate(str(i),xytext=(comcast_daily_complains_data.index[index],i),xy=(comcast_daily_complains_data.index[index],i)) #xytext will define position of comments

    index=index+1    

plt.legend() 
comcast_data['Month'] = comcast_data.Date.dt.month

comcast_monthly_complains_data = pd.DataFrame({'count':comcast_data.groupby('Month')['Ticket #'].count()})

comcast_monthly_complains_data.sort_values(by='Month', ascending = True)

comcast_monthly_complains_data

plt.figure(figsize=(24,6))

plt.title("Monthly Comcast Telecom Consumer Complaints")

plt.xlabel("Month of complaint")

plt.ylabel("Number of complaints")

plt.plot(comcast_monthly_complains_data)

index=0

for i in comcast_monthly_complains_data['count'].values.tolist():

    plt.annotate(str(i),xytext=(comcast_monthly_complains_data.index[index],i),xy=(comcast_monthly_complains_data.index[index],i)) #xytext will define position of comments

    index=index+1   

plt.show()
def transform_status(row):

    if((row.Status == 'Open') | (row.Status == 'Pending')):

        return "Open"

    elif((row.Status == 'Closed') | (row.Status == 'Solved')):

        return "Closed"

    else:

        return "Unknown"

comcast_data['Complain Status'] = comcast_data.apply(transform_status,axis='columns')

comcast_statewise_complains_data = pd.DataFrame({'Total':comcast_data.groupby(['State','Complain Status'])['Complain Status'].count()})

comcast_statewise_complains_data = comcast_statewise_complains_data.unstack()

comcast_statewise_complains_data = comcast_statewise_complains_data.fillna(0)

plt.figure(figsize=(24,20))

open_plt = plt.barh(comcast_statewise_complains_data.index,comcast_statewise_complains_data['Total','Open'] ,tick_label=comcast_statewise_complains_data.index)

closed_plt = plt.barh(comcast_statewise_complains_data.index,comcast_statewise_complains_data['Total','Closed'], left = comcast_statewise_complains_data['Total','Open'])

plt.legend((open_plt[0], closed_plt[0]), ('Open', 'Closed'))
comcast_statewise_complains_data['Resolved Complaints'] = (comcast_statewise_complains_data['Total','Closed'])/(comcast_statewise_complains_data['Total','Closed']+comcast_statewise_complains_data['Total','Open'])

comcast_statewise_complains_data['Resolved Complaints'].mean()