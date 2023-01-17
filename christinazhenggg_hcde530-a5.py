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

import datetime as dt

import pandas as pd
dc = pd.read_csv("../input/pddataset/filmdeathcounts.csv")

#dc.set_index('Year', inplace=True)

#dc.sort_index(axis=0, ascending=False)

dc
# create a scatter chart to see how movie body count is distributed over the years

colors = ['#ffa64d']

dc.plot(kind='scatter',x='Year',y='Body_Count', colors=colors, figsize=(14, 9), alpha=0.6, s=40)
# define body count # ranges

def bodycount(death):

    if death < 100:

        return 'Below 100' 

    elif death > 100 and death <= 200:

        return 'From 100 to 200'

    elif death > 200:

        return 'Above 200'



# create a pie chart that shows the ratio for each range

colors = ['#0099cc', '#ffcc00','#ffa64d']

dc['Body_Count'].apply(bodycount).value_counts().plot(kind='pie', startangle=90,figsize=(8, 8), fontsize=10, autopct='%.1f%%', colors=colors, explode = (0.1,0,0), title='Body Counts in Movies (1947-2013)')
# create a new df that only has movies with about 8.5 IMDB rating

rating = dc[dc['IMDB_Rating'] >= 8.5 ]

rating.set_index('Film', inplace=True)

rating



# plot the df into a horizontal bar chart.

colors2 = ['#0099cc']

rating[['Body_Count']].plot(kind='barh',figsize=(12,8), alpha=0.9, title = 'Good Movie Body Counts', colors= colors2)