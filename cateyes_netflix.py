# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")

data.head()
data.info()
data.describe()
data.columns
movie = data['type'].value_counts()



movie


data.type=[ each.lower() for each in data.type]

data.type=[each.split()[0]+"_"+each.split()[1] if(len(each.split())>1) else each for each in data.type]

data.type

movie = data[data.type == "movie"]

tv_show = data[data.type == "tv_show"]

plt.hist(movie.type,bins= 1)

plt.hist(tv_show.type,bins= 1)

plt.xlabel("Type")

plt.ylabel("Count")

plt.title("Movie and TV Show")

plt.show()
data_year = data['release_year'].value_counts()

data_year=pd.DataFrame(data_year).reset_index()

data_year.head(10)



plt.figure(figsize=(12,6))

plt.hist(data.release_year,color='pink' ,bins= 100)

plt.xlabel("Year")

plt.ylabel("Count")

plt.title("Publication Counts by Years")

plt.show()

plt.figure(figsize=(12,6))

country_with_movie= data.country.value_counts()[:10]

pd.Series(country_with_movie).sort_values(ascending=True).plot.barh(width=0.9, color='purple')

plt.title("How many movies are in which countries")

plt.show()
plt.figure(figsize=(12,6))

data[data["type"]=="movie"]["listed_in"].value_counts()[:10].plot(kind="barh",color="yellow")

plt.title("Top 10 Genres of Movies",size=18)

plt.show()
plt.figure(figsize=(12,6))

data[data["type"]=="tv_show"]["listed_in"].value_counts()[:10].plot(kind="barh",color="red")

plt.title("Top 10 Genres of TV Show",size=18)

plt.show()