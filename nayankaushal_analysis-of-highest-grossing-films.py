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
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
data = pd.read_csv("/kaggle/input/top-10-highest-grossing-films-19752018/blockbusters.csv")

data.head()
data['worldwide_gross'] = data['worldwide_gross'].str.replace('$', '').str.replace(',', '').astype(float)

data.head()
data.isna().sum()
data = data.fillna('No Sub Genre')
data.head()
highest_grossing = data.groupby('Main_Genre').sum()['worldwide_gross'].reset_index().sort_values(by='Main_Genre', ascending=False)

display(highest_grossing)

print("Highest Grossing Main Genre is: ")

display(highest_grossing[highest_grossing.worldwide_gross == highest_grossing.worldwide_gross.max()])
count_title = data.groupby('Main_Genre').count().sort_values(by='Main_Genre', ascending=False)['title']

count_title
np_highest_grossing = np.array(highest_grossing.worldwide_gross)

np_count_title = np.array(count_title)

np_index = np.array(count_title.index)

np_average_amount_per_genre = np_highest_grossing/np_count_title
np_result = np.vstack((np_index, np_average_amount_per_genre)).transpose()

result = pd.DataFrame({'Genre' : np_result[: , 0], 'Average Amount Grossed' : np_result[:, 1]})

result
result.loc[result['Average Amount Grossed'] == result['Average Amount Grossed'].max()]
plt.figure(figsize = (20,7))

sns.set_style("darkgrid")

sns.set_palette("PRGn")

sns.catplot(x="Main_Genre", y="worldwide_gross", data=data, kind="bar", estimator=sum, ci=None)

plt.xticks(rotation=90)

plt.title('Film Genres with their worldwide gross sum')

plt.xlabel('Main Genre')

plt.ylabel('Worldwide Gross Sum')
plt.figure(figsize = (20,7))

sns.set_style("darkgrid")

sns.catplot(x="Main_Genre", data=data, kind="count")

plt.xticks(rotation=90)

plt.title('Count of Films in each Genre')

plt.xlabel('Main Genre')

plt.ylabel('No. of Films')
plt.figure(figsize=(20,7))

sns.set_style("darkgrid")

sns.catplot(x='Genre', y='Average Amount Grossed', data=result, kind='bar')

plt.xticks(rotation=90)

plt.title('Average Amount grossed per film')

plt.xlabel('Main Genre')

plt.ylabel('Average Amount Grossed per film')
film = data.loc[:, ['Main_Genre','title', 'imdb_rating', 'worldwide_gross']].sort_values(by='imdb_rating')

film
plt.figure(figsize=(10,3))

sns.set_style("darkgrid")

g=sns.lineplot(x='imdb_rating', y='worldwide_gross', data=film, ci=None)

g.set_title("IMDB Ratings vs Worldwide Gross")

g.set(xlabel="IMDB Rating", ylabel="Worldwide Gross")
plt.figure(figsize=(30,7))

sns.set(style="ticks")

sns.jointplot(x=film.imdb_rating, y=film.worldwide_gross, kind= 'reg', color = '#4CB391').set_axis_labels("IMDB Ratings", "Worldwide Gross")
data_pivot_table = data.pivot_table(index='Main_Genre', values='worldwide_gross', columns = 'rating', fill_value = 0, margins=True)

data_pivot_table
max_avg_gross = data_pivot_table.loc['All']

max_avg_gross[max_avg_gross == max_avg_gross.max()]
plt.figure(figsize=(5,5))

sns.set_style("darkgrid")

sns.set_palette("PRGn")

sns.barplot(x=data.rating, y=data.worldwide_gross, ci=None)

plt.xlabel('Rating')

plt.ylabel('Worldwide Gross')

plt.title('Rating vs. Worldwide Gross')