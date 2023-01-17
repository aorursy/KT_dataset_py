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
#Importing the required Libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
#Reading the Netflix csv file using Pandas

df = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')

df.head() #Displaying few first few lines of the data
import missingno as miss



miss.matrix(df)
#Removing rows having no Title specified 

df = df[df['title'].notna()]



#Filling NAN values

df['director'] = df['director'].fillna('Unknown')



df['cast'] = df['cast'].fillna('Unknown')



df['country'] = df['country'].fillna('Not Mentioned')
#Only TV Shows

TV = df[df['type'] == 'TV Show']



#Only Movies

movie = df[df['type'] == 'Movie']



#Sorting rows in descending order according to released year and Data added

TV = TV[TV['release_year'] >= 2015].sort_values(by = ['release_year', 'date_added'], ascending = False)

movie = movie[movie['release_year'] >= 2015].sort_values(by = ['release_year', 'date_added'], ascending = False)
import seaborn as sns



plt.figure(figsize = (7, 5))

sns.set(style = 'darkgrid')

sns.countplot(df['type'][df.release_year >= 2015])



plt.show()
#Directors with atleast one Movie OR atleast one TV Show

TVorMovie = df[['type', 'director']]

TVorMovie = TVorMovie.pivot_table(index = ['director'], columns = ['type'], aggfunc = len).fillna('...')

TVorMovie.head(10)
#Directors with at least one Movie AND at least one TV Show.

TVnMovie = TVorMovie[TVorMovie['TV Show'] != '...']

TVnMovie[TVnMovie['Movie'] != '...']
ratingCount = df[['type', 'rating']]

ratingCount = ratingCount.groupby('rating').count()



plt.figure(figsize = (15, 5))

bars = plt.bar(ratingCount.index, ratingCount['type'], color = 'r')

plt.tick_params(axis = 'both', left = False, bottom = False)

plt.gca().spines['right'].set_visible(False)

plt.gca().spines['top'].set_visible(False)

plt.gca().spines['left'].set_visible(False)

plt.gca().spines['bottom'].set_visible(False)

plt.tick_params(labelleft = False)



plt.title('Rating Count', fontsize = 25)



for i in bars:    

      plt.gca().text(i.get_x() + i.get_width()/2, i.get_height() + 7, 

      str(int(i.get_height())), ha = 'center', color = 'k')

plt.show()
US_Based = df[['type', 'title', 'country', 'rating']].copy()

US_Based['country'] = US_Based['country'].str.extract(r'(United States)+')

US_Based.dropna().sort_values(['type','title']).reset_index(drop = True).head(10)
yearCount = df[df['release_year'] > 2010]

yearCount = yearCount.groupby('release_year')['release_year'].apply(len)



plt.figure(figsize = (12, 8))

bars = plt.bar(yearCount.index.tolist(), yearCount.values.tolist(), color = 'grey')



plt.tick_params(axis = 'both', left = False, bottom = False)

plt.gca().spines['right'].set_visible(False)

plt.gca().spines['top'].set_visible(False)

plt.gca().spines['left'].set_visible(False)

plt.gca().spines['bottom'].set_visible(False)

plt.tick_params(labelleft = False)



plt.xticks(yearCount.index.tolist())

plt.title('No. of Movie and TV Show Released Every Year', fontsize = 25)



bars[7].set_color('black')

for i in bars:    

      plt.gca().text(i.get_x() + i.get_width()/2, i.get_height() + 7, 

      str(int(i.get_height())), ha = 'center', color = 'k')



plt.show()