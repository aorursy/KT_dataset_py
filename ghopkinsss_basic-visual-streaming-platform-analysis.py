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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

%matplotlib inline
video_data = pd.read_csv('/kaggle/input/tv-shows-on-netflix-prime-video-hulu-and-disney/tv_shows.csv')
display(video_data.head())
video_data = video_data.drop(['Unnamed: 0', 'type'], axis=1)
display(video_data.info())
duplicates = video_data.duplicated(subset='Title', keep=False)
duplicated_values = video_data[duplicates].sort_values('Title')
display(duplicated_values.head(20))
netflix_sum = video_data['Netflix'].sum()
hulu_sum = video_data['Hulu'].sum()
prime_sum = video_data['Prime Video'].sum()
disney_sum = video_data['Disney+'].sum()

market_shares = [netflix_sum, hulu_sum, prime_sum, disney_sum]
labels = ('Netflix', 'Hulu', 'Amazon Prime', 'Disney+')
sep = (0.05, 0.05, 0.05, 0.05)

fig, ax = plt.subplots()

ax.pie(market_shares,
      explode=sep,
      labels = labels,
      autopct = '%.1f%%',
      radius=2,
     textprops={'fontsize': 14})

plt.show()

print('Netflix offers ' + str(netflix_sum) + ' tv-shows')
print('Hulu offers ' + str(hulu_sum) + ' tv-shows')
print('Prime offers ' + str(prime_sum) + ' tv-shows')
print('Disney+ offers ' + str(disney_sum) + ' tv-shows')
#Dataframes for each platform
netflix_shows = video_data.loc[video_data['Netflix'] == 1].drop(['Hulu', 'Prime Video', 'Disney+'], axis=1)
hulu_shows = video_data.loc[video_data['Hulu'] == 1].drop(['Netflix', 'Prime Video', 'Disney+'], axis=1)
prime_shows = video_data.loc[video_data['Prime Video'] == 1].drop(['Hulu', 'Netflix', 'Disney+'], axis=1)
disney_shows = video_data.loc[video_data['Disney+'] == 1].drop(['Hulu', 'Prime Video', 'Netflix'], axis=1)

#Who is the platform with the most highly rated shows?
top_netflix_shows = netflix_shows['IMDb'] > 8
top_hulu_shows = hulu_shows['IMDb'] > 8 
top_prime_shows = prime_shows['IMDb'] > 8
top_disney_shows = disney_shows['IMDb'] > 8

top_rated_shows = pd.DataFrame({
    'Platforms' : ['Netflix', 
                   'Hulu', 
                   'Prime Video', 
                   'Disney'],
    'Total highly rated shows' : [top_netflix_shows.sum(),
                                  top_hulu_shows.sum(),
                                  top_prime_shows.sum(),
                                  top_disney_shows.sum()]})
sns.barplot(x = top_rated_shows['Platforms'],
            y = top_rated_shows['Total highly rated shows'],
            data = top_rated_shows,
           palette = 'bright')

plt.xlabel('Platforms')
plt.ylabel('Total shows with 8+ Ratings')
plt.title('Platforms with Highly Rated Shows')
plt.show()

top_rated_shows = top_rated_shows.set_index('Platforms')
display(top_rated_shows)
#Who is the platform with the best concentration of highly rated shows? 

ratio_netflix = (top_netflix_shows.sum() / netflix_sum * 100)
ratio_hulu = (top_hulu_shows.sum() / hulu_sum * 100)
ratio_prime = (top_prime_shows.sum() / prime_sum * 100)
ratio_disney = (top_disney_shows.sum() / disney_sum * 100)

top_rated_ratios = ({
    'platforms' : ['Netflix',
                   'Hulu',
                   'Prime Video',
                  'Disney+'],
    'ratios' : [ratio_netflix,
               ratio_hulu,
               ratio_prime,
               ratio_disney]
})

sns.barplot(x = top_rated_ratios['platforms'],
           y = top_rated_ratios['ratios'],
           palette = 'colorblind')

plt.xlabel('Platforms')
plt.ylabel('Number of highly rated tv-shows per 100')
plt.title('High Quality Show Concentration')

plt.show()

print('Netflix provides on average ' + str(int(round(ratio_netflix))) + ' high quality tv-shows out of every 100 tv-shoows')
print('Hulu provides on average ' + str(int(round(ratio_hulu))) + ' high quality tv-shows out of every 100 tv-shoows')
print('Prime Video provides on average ' + str(int(round(ratio_prime))) + ' high quality tv-shows out of every 100 tv-shoows')
print('Disney+ provides on average ' + str(int(round(ratio_disney))) + ' high quality tv-shows out of every 100 tv-shoows')
#Best tv-shows on offer for each streaming platform according to IMDb

netflix_top_10 = netflix_shows.sort_values(by='IMDb', ascending = False).head(10)

plt.figure(figsize=(8,4))
sns.barplot(x='IMDb', y='Title', data=netflix_top_10, palette='bright')

plt.xlabel('IMDb Score')
plt.ylabel('TV Show')
plt.title('Highest Rated Netflix Shows')

plt.show()
hulu_top_10 = hulu_shows.sort_values(by='IMDb', ascending = False).head(10)

plt.figure(figsize=(8,4))
sns.barplot(x='IMDb', y='Title', data=hulu_top_10, palette='bright')

plt.xlabel('IMDb Score')
plt.ylabel('TV Show')
plt.title('Highest Rated Hulu Shows')

plt.show()
prime_top_10 = prime_shows.sort_values(by='IMDb', ascending = False).head(10)

plt.figure(figsize=(8,4))
sns.barplot(x='IMDb', y='Title', data=prime_top_10, palette='bright')

plt.xlabel('IMDb Score')
plt.ylabel('TV Show')
plt.title('Highest Rated Prime Shows')

plt.show()
disney_top_10 = disney_shows.sort_values(by='IMDb', ascending = False).head(10)

plt.figure(figsize=(8,4))
sns.barplot(x='IMDb', y='Title', data=disney_top_10, palette='bright')

plt.xlabel('IMDb Score')
plt.ylabel('TV Show')
plt.title('Highest Rated Disney+ Shows')

plt.show()
#Who has access to the top 10 tv-shows currently?

video_data['platform'] = video_data[['Netflix', 'Hulu', 'Prime Video', 'Disney+']].idxmax(axis=1)

overall_top_10 = video_data.sort_values(by='IMDb', ascending=False).head(10)

plt.figure(figsize=(8,8))
sns.catplot(x='IMDb', y='Title', data=overall_top_10,hue='platform', kind='bar')
plt.show()
#Which age grouping exist?
print(video_data.Age.unique())
#'all' signifies they are suitable for children, which platforms are best subscribed to for children entertainment?

net_child = netflix_shows[netflix_shows['Age'] == 'all']
hulu_child = hulu_shows[hulu_shows['Age'] == 'all']
prime_child = prime_shows[prime_shows['Age'] == 'all']
disney_child = disney_shows[disney_shows['Age'] == 'all']

net1 = len(net_child['Title'])
hulu1 = len(hulu_child['Title'])
prime1 = len(prime_child['Title'])
disney1 = len(disney_child['Title'])

children_sum = ({
    'platforms' : ['Netflix',
                   'Hulu',
                   'Prime Video',
                  'Disney+'],
    'sums' : [net1,
               hulu1,
               prime1,
               disney1]
})
plt.figure(figsize=(10,10))
sns.barplot(x = children_sum['platforms'],
           y = children_sum['sums'],
           palette = 'colorblind')

plt.xlabel('Platforms')
plt.ylabel('Number of TV-Shows Suitable for Children')
plt.title('Number of TV-shows Suitable for Children per Platform')

plt.show()

print('Netflix has ' + str(net1) + ' tv-shows for children')
print('Hulu has ' + str(hulu1) + ' tv-shows for children')
print('Prime has ' + str(prime1) + ' tv-shows for children')
print('Disney+ has ' + str(disney1) + ' tv-shows for children')
ratio1_netflix = net1 / netflix_sum
ratio1_hulu = hulu1 / netflix_sum
ratio1_prime = prime1 / prime_sum
ratio1_disney = disney1 / disney_sum

children_show_ratios = ({
    'platforms' : ['Netflix',
                   'Hulu',
                   'Prime Video',
                  'Disney+'],
    'ratios' : [ratio1_netflix,
               ratio1_hulu,
               ratio1_prime,
               ratio1_disney]
})

plt.figure(figsize=(10,10))
sns.barplot(x = children_show_ratios['platforms'],
           y = children_show_ratios['ratios'],
           palette = 'colorblind')

plt.xlabel('Platforms')
plt.ylabel('Ratio of Child Suitable TV-Shows')
plt.title('Ratio of Child Suitable TV-Shows per Platform')

plt.show()

print(ratio1_disney)