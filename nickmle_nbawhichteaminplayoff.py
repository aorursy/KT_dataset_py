# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

import matplotlib as plt

import time



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
nbadata = pd.read_csv('../input/nba.games.stats.csv')
from sklearn.preprocessing import StandardScaler



'''

2 centered clustering 2 classification will work very well, I can use any method that is useful.



'''

#datetime.datetime.strftime(cledata['Date'], '%Y-%m-%d') > datetime.datetime.strptime('20150630', '%Y%m%d').date()

nbadata.head()

teams = list(nbadata['Team'].unique())
nbadata.describe()
# Data collections related to nba playoff predictions and 



# Web scraping  -- automating the process of extrating information from the web pages

# for data collection and analysis

# for incorporating in a web app



# browser --> server --> program -->



'''

Always against the Terms of Use of a Website



factual, non-proprietary data is generally ok;

proprietary data scraping depends on what you do with it.

potential or actual damage to the scrapee(denial of service)

public vs. private information

purpose

try to get the information openly

Is there a public interest involved ..



Libraries for webscraping # Using api far better

request: handles http: request and responses

Beautiful Soup: utilizes the tag structure of an html page to quickly parse contents of a page adn retireve

Selenium : emulates a browser. Useful when a page contains scripts



    --this is independent API but python has support library for it .. 

    want to be able to dig multiple branches of big tree structures ....

    

'''





#import requests

#from bs4 import BeautifulSoup

#url = 'http://www.espn.com/nba/scoreboard/_/date/20190413'

#response = requests.get(url)#

#print(response.status_code)



'''

Basically scraping the data is not so efficient specially data is real time or live changing. 



Using webscraper or api is the one that may be useful to use prevalently .. 



'''

print()
"""Note will be updated and get completed soon"""