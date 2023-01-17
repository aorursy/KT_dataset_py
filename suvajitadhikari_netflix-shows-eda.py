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

netflix_shows=pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")

netflix_shows.head(10)
netflix_shows.shape              # let's see how big is the dataset
# check null values of different columns

netflix_shows.isnull().sum()
# Analyse counts of different ratings  

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style("darkgrid")

plt.figure(figsize=(14,8))

sns.countplot(x="rating",data=netflix_shows)
netflix_shows["date_added"]=netflix_shows["date_added"].fillna(method='ffill')

netflix_shows["date_added"].isnull().sum()     
year=netflix_shows["date_added"].to_list()  # convert to list for extracting the year added
for i in range(len(year)):

    netflix_shows["year_added"].iloc[i]=year[i].split(',')[1]
netflix_shows.head()
import numpy as np

netflix_shows["year_added"]=netflix_shows["year_added"].astype(np.int)
sns.set_style("darkgrid")

plt.figure(figsize=(14,8))

sns.countplot(x="type",hue="year_added",data=netflix_shows)
import matplotlib.pyplot as plt

values=dict(netflix_shows["type"].value_counts())

labels=values.keys()

sizes=values.values()

fig1, ax1 = plt.subplots()

ax1.pie(sizes,labels=labels, shadow=True,autopct='%1.1f%%',startangle=90)

plt.show()
# Let's check some oldest movies and TV shows on netflix

netflix_shows[["title","release_year"]].groupby(["release_year"]).min().head(10)

# Let's focus on the distribution of duration of movies

def plot_movie_duration():

    netflix_duration=netflix_shows["duration"].copy()

    index=[]

    for i in range(len(netflix_shows["duration"])):

        if netflix_shows["type"].iloc[i]=='TV Show':

            index.append(i)

        

        

    netflix_duration=netflix_duration.drop(index)

    for i in range(len(netflix_duration)):

        x=netflix_duration.iloc[i].split(' ')[0]

        netflix_duration.iloc[i]=x

        

    plt.figure(figsize=(12,8))

    sns.distplot(netflix_duration)



plot_movie_duration()
def plot_series_duration():

    netflix_tv_show_duration=netflix_shows["duration"].copy()

    index=[]

    for i in range(len(netflix_shows["duration"])):

        if netflix_shows["type"].iloc[i]=='Movie':

            index.append(i)

        

        

    netflix_tv_show_duration=netflix_tv_show_duration.drop(index)

    for i in range(len(netflix_tv_show_duration)):

        x=netflix_tv_show_duration.iloc[i].split(' ')[0]

        netflix_tv_show_duration.iloc[i]=x

        

    plt.figure(figsize=(12,8))

    sns.countplot(netflix_tv_show_duration)



plot_series_duration()
# Now see the top categories of shows

from collections import Counter 





Genres=netflix_shows["listed_in"].copy()

categories=[]

for i in range(len(Genres)):

    elements=Genres.iloc[i].split(",")

    for j in range(len(elements)):

        categories.append(elements[j])

count_categories=Counter(categories)

#print(count_categories)



count_categories=count_categories.most_common()   # for sorting values

keys=[count_categories[i][0] for i in range(len(count_categories))]

values=[count_categories[i][1] for i in range(len(count_categories))]





top_10_keys=keys[0:10]

top_10_values=values[0:10]

plt.figure(figsize=(21,10))

sns.barplot(top_10_keys,top_10_values)

plt.title("Top 10 movie-series categories")
country=netflix_shows["country"].copy()

country=country.dropna(axis=0,how='all')

country_list=[]

for i in range(len(country)):

    elements=country.iloc[i].replace(" ","").split(",")

    for j in range(len(elements)):

        country_list.append(elements[j])

country_dict=Counter(country_list)

#print(country_dict)

country_dict=country_dict.most_common()



keys=[country_dict[i][0] for i in range(len(country_dict))]

values=[country_dict[i][1] for i in range(len(country_dict))]



top_20_countries=keys[:20]

top_20_values=values[:20]



plt.style.use('Solarize_Light2')

plt.figure(figsize=(14,10))

sns.barplot(top_20_values,top_20_countries)
actors_country=netflix_shows[['cast','country']]

actors_country=actors_country.dropna(axis=0,how='any')



top_actors_USA=[]

top_actors_India=[]

top_actors_UK=[]

top_actors_Canada=[]

top_actors_France=[]



for i in range(actors_country.shape[0]):

    country = list(actors_country['country'].iloc[i].replace(" ", "").split(","))

    actors= list(actors_country['cast'].iloc[i].split(","))

    for j in range(len(actors)):

        for k in range(len(country)):

            actors[j]=actors[j].strip()

            if country[k]=='UnitedStates':

                top_actors_USA.append(actors[j])

            if country[k]=='India':

                top_actors_India.append(actors[j])

            if country[k]=='UnitedKingdom':

                top_actors_UK.append(actors[j])

            if country[k]=='Canada':

                top_actors_Canada.append(actors[j])

            if country[k]=='France':

                top_actors_France.append(actors[j])



def plot_top_actors(top_actors):

    top_actors=Counter(top_actors)

    top_actors=top_actors.most_common()

    

    keys=[top_actors[i][0] for i in range(len(top_actors))]

    values=[top_actors[i][1] for i in range(len(top_actors))]



    top_20_keys=keys[:20]

    top_20_values=values[:20]

    

    plt.style.use('classic')

    plt.figure(figsize=(10,6))

    sns.barplot(top_20_values,top_20_keys)

    



plot_top_actors(top_actors_India)

plt.title("Top Indian actors")



#plot_top_actors(top_actors_USA)
plot_top_actors(top_actors_USA)

plt.title("Top American actors")
plot_top_actors(top_actors_UK)

plt.title("Top English actors")
plot_top_actors(top_actors_Canada)

plt.title("Top Canadian actors")
plot_top_actors(top_actors_France)

plt.title("Top French actors")
directors_country=netflix_shows[['director','country']]

directors_country=directors_country.dropna(axis=0,how='any')



top_director_USA=[]

top_director_INDIA=[]

top_director_UK=[]

top_director_CANADA=[]

top_director_FRANCE=[]



for i in range(directors_country.shape[0]):

    country = list(directors_country['country'].iloc[i].replace(" ", "").split(","))

    directors= list(directors_country['director'].iloc[i].split(","))

    for j in range(len(directors)):

        for k in range(len(country)):

            directors[j]=directors[j].strip()

            if country[k]=='UnitedStates':

                top_director_USA.append(directors[j])

            if country[k]=='India':

                top_director_INDIA.append(directors[j])

            if country[k]=='UnitedKingdom':

                top_director_UK.append(directors[j])

            if country[k]=='Canada':

                top_director_CANADA.append(directors[j])

            if country[k]=='France':

                top_director_FRANCE.append(directors[j])

def plot_top_directors(top_directors):

    top_directors=Counter(top_directors)

    top_directors=top_directors.most_common()

    

    keys=[top_directors[i][0] for i in range(len(top_directors))]

    values=[top_directors[i][1] for i in range(len(top_directors))]



    top_20_keys=keys[:20]

    top_20_values=values[:20]

    

    plt.style.use('classic')

    plt.figure(figsize=(10,6))

    sns.barplot(top_20_values,top_20_keys)

    



plot_top_actors(top_director_INDIA)

plt.title("Top Indian directors")

plot_top_actors(top_director_USA)

plt.title("Top American directors")
plot_top_actors(top_director_UK)

plt.title("Top English directors")
plot_top_actors(top_director_CANADA)

plt.title("Top Canadian directors")
plot_top_actors(top_director_FRANCE)

plt.title("Top French directors")