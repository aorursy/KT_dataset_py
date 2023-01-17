# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# reading the csv file

df = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
# lets see the head of this dataset

len(df)
# how many nulls i have in my dataset?

df.isnull().sum()
# i want to know how many movies were added and realeaded in the same year

# at first it's important to remove the nulls

df = df.dropna(subset=['date_added'])
df.isnull().sum()
# implementing a simple way to count what we want

df['date_added'] = int(df['date_added'][0].split(',')[1].replace(' ', ''))
# Ok, how about a histogram? Lets do this!

plt.hist(df['date_added'] - df['release_year'])

plt.show()
# Oh my! There is a diference between added date and release date of 80 years!  

df[(df['date_added'] - df['release_year']) > 80]
# how many movies were added and realeaded in the same year?

df[(df['date_added'] - df['release_year']) == 0].info()
# Nice, 843 were realeased and added in the same year!

# I have to watch this 20's TV Show! Bye bye guys, i hope you enjoyed this notebook!