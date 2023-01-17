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



%matplotlib inline
df = pd.read_csv('/kaggle/input/chess-fide-ratings/players.csv')

df.head()
title_values = df['title'].value_counts().reset_index()

title_values

fig = plt.figure(figsize=(12,6))

sns.barplot(x ='index',y = 'title',data = title_values)

plt.xlabel('Title_names')

plt.tight_layout()
rat = pd.read_csv('/kaggle/input/chess-fide-ratings/ratings_2020.csv')

rat.head(10)
data = pd.merge(df,rat,on ='fide_id')

data.head(10)
ratings = data[['name','rating_standard','month']][data['month']==2]

ratings.fillna(0,inplace = True)

ratings.head()
fig = plt.figure(figsize = (18,8))

sns.barplot(x = 'name',y = 'rating_standard',data = ratings.head(10))

plt.ylim([0,3500])

plt.xlabel('Player Names')

plt.ylabel('Ratings')

plt.title('Top 10 players in first month of 2020')

fig.tight_layout()
rating = data[['name','rating_standard','month']]

rating.head(10)
fig = plt.figure(figsize = (18,8))

sns.barplot(x = 'name' ,y ='rating_standard',data = rating.head(30),hue = 'month')

plt.ylim([1550,2550])

plt.xlabel('Player Names')

plt.ylabel('Ratings')

plt.title('Top 5 player ratings of 2020 in all months  ')