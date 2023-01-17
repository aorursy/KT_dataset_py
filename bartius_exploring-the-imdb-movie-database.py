# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from bs4 import BeautifulSoup
import requests
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
movies = pd.read_csv('../input/movies.csv')
movies = movies[movies.columns[1:]]
movies.head()
movies['profit'] = movies.apply(lambda row: row.cum_worldwide_gross - row.budget, axis=1)
movies['year'] = movies.apply(lambda row: row.release_date.split()[-1], axis=1)
movies.head()
ratings = movies.groupby('rating').count()['title']
plt.pie(ratings, autopct='%1.1f%%', labels=ratings.index.values)
plt.title('Percentage of movies that fall into each rating category')
plt.show()
isnull = movies[movies['rating'].isnull()]
total = movies['rating']
print(len(isnull)/len(total))
# More than half of the columns are empty!
movies = movies.dropna(subset=['release_date'])
profit_correction = list()
for row in movies.iterrows():
    html = requests.get('http://www.in2013dollars.com/'+row[1]['year']+'-dollars-in-2018?amount=1').text
    soup = BeautifulSoup(html, 'lxml')
    profit_correction.append(float(soup.find('h1', {'style': 'text-align: center; line-height: 1.5'}).findAll('span')[1].text.split('$')[1])*row[1]['profit'])
    
movies['profit_correction'] = profit_correction
movies.head()
profit = movies.groupby('year').mean()
profit = profit.dropna(subset=['profit_correction'])
profit = profit[['profit_correction']]

num_per_year = movies.groupby('year').count()
num_per_year = num_per_year[['title']]
num_per_year.columns = ['num_made']

x = profit.reset_index().merge(num_per_year.reset_index())
x['num_made'] = x.apply(lambda row: row.num_made*100, axis=1)
plt.figure(figsize=(25,15))
plt.xticks(rotation=45)
plt.scatter(x=x['year'], y=x['profit_correction'], s=x['num_made'], c=x['num_made'])
plt.colorbar(label='Number of movies made multiplied by 100')
plt.title('Profits of movies over time')
plt.ylabel('Profit')
#It looks like more movies that made the list were filmed fairly recently. However, the profitability seems to have remained the same, if not becoming lower.¶
runtime = movies.dropna(subset=['runtime'])['runtime']
plt.hist(runtime, bins=20, color='blue')
plt.xlabel('Runtime')
plt.ylabel('Number of Movies')
plt.title('Histogram of the number of movies that fall into certain runtimes')
#Although it looks like 100 to 150 minutes is the sweet spot for making the list of top movies, there is no comparison to the list of bottom movies. So no conclusion can be drawn.
movies['lead'] = movies.apply(lambda row: row.cast.split(';')[0], axis=1)
leads = movies.groupby('lead').count()
leads = leads[leads['title'] >1]
leads = leads[leads.columns[0]]
ax = leads.plot(kind='bar', figsize=(20, 6), rot=90)
ax.set_ylabel('Number of movies')
ax.set_title('The number of movies each actor has been cast')
#It looks like Leonardo DiCaprio, Robert DeNiro and Tom Hanks have made more movies that landed on the most popular list than any other actor/actress!