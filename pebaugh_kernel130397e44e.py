## PETER EBAUGH - DATA SCIENCE FOR BUSINESS



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
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



steam = pd.read_csv('/kaggle/input/steam-store-games/steam.csv')

##steam.head(5)

plt.hist(steam.price.dropna(), 30, range=[0, 60], facecolor='gray', align='mid')

plt.title('Price Distribution of Games')

plt.xlabel("Price")

plt.ylabel("# of games (frequency)")

plt.show()
# Get current size

fig_size = plt.rcParams["figure.figsize"]

# Prints: [8.0, 6.0]

print("Current size:", fig_size)

# Set figure width to 12 and height to 9

fig_size[0] = 45

fig_size[1] = 25

plt.rcParams["figure.figsize"] = fig_size



new = pd.DataFrame(steam.genres.str.split(';').tolist()).stack()

##new = new.reset_index()[[0, 'var2']] 

##b.columns = ['var1', 'var2']

##new = steam["genres"].str.split(";", n = -1, expand = True) 

##for x in range(1, 16):

    ##new.iloc[:,0].append(new.iloc[:,x])



new = new.to_frame().reset_index()

new.rename(columns = {0:'genres'}, inplace = True)

new.drop('level_0', axis=1, inplace=True)

new.drop('level_1', axis=1, inplace=True)

##print(new.head(26))



count_by_genres = new.groupby('genres').size()

plt.bar(count_by_genres.index.tolist(), count_by_genres.tolist())

plt.title('Steam Genres')

plt.ylabel("Total #")







plt.show()
steam['release_month'] = pd.DatetimeIndex(steam['release_date']).month



count_by_month = steam.groupby('release_month').size()

plt.bar(count_by_month.index.tolist(), count_by_month.tolist())

plt.title('Release Month')

plt.ylabel("Total Games")

plt.show()

steam['release_year'] = pd.DatetimeIndex(steam['release_date']).year



count_by_year = steam.groupby('release_year').size()

plt.bar(count_by_year.index.tolist(), count_by_year.tolist())

plt.title('Release Year')

plt.ylabel("Total Games")

plt.show()
steam[steam.required_age > 16].head(6)