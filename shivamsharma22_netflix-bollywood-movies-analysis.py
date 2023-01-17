# Import Important Python Libraries



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import re
# Loading of Data

dframe = pd.read_csv("../input/netflix-shows/netflix_titles_nov_2019.csv")
dframe.head()
# Filtering of Bollywood(Indian) Data from the whole dataframe



dframe_india = dframe[dframe['country']=='India']
dframe_india.shape
dframe_india['type'].unique()
dframe_india = dframe_india[dframe_india['type']=='Movie']
dframe_india.shape
# removal of unwanted columns



dframe_india.drop(['country','type'],axis=1,inplace=True)
dframe_india.head()
# Creating a dictionary based on rating values



Movie_list = {'TV-Y7':'Child Movies',

              'TV-G':'Family Movies',

              'TV-PG':'Family Movies-Parental Guidance',

              'TV-14':'Family Movies-Parental Guidance',

              'TV-MA':'Adult Movies','TV-Y7-FV':'Child Movies',

              'PG-13':'Family Movies-Parental Guidance',

              'PG':'Family Movies-Parental Guidance',

              'R':'Adult Movies',

              'NR':'Unrated Movies',

              'UR':'Unrated Movies'}
# Adding a column named MOVIE TYPE in the existing Dataframe based on the mapping with dictionary values 



dframe_india['Movie Type'] = dframe_india['rating'].map(Movie_list)
dframe_india.head()
sns.countplot(y='Movie Type',data=dframe_india,palette='Set1')
# List of Child Movies in this Dataframe



dframe_india[dframe_india['Movie Type']=='Child Movies'].head(5)
# List of Adult Movies in this Dataframe



dframe_india[dframe_india['Movie Type']=='Adult Movies'].head(5)
# List of Family Movies with little Parental Guidance in this Dataframe



dframe_india[dframe_india['Movie Type']=='Family Movies-Parental Guidance'].head(5)
# List of Family Movies suitable for all ages in this Dataframe



dframe_india[dframe_india['Movie Type']=='Family Movies'].head(5)
# Creating a function to separate movies in different categories based on listed_in column present in this Dataframe



def category_separator(category,show_id):

    for i in (re.split(r',',category)):

        if i.strip() in dframe_india:

            dframe_india[i.strip()][dframe_india['show_id']==show_id]='YES'

        else:

            dframe_india[i.strip()]='NO'

            dframe_india[i.strip()][dframe_india['show_id']==show_id]='YES'
# Calling of category_separator function. This Loop will run for all the rows in the Dataframe and will separate all the movies in different categories.



for show_id, category in zip(dframe_india.show_id, dframe_india.listed_in): 

    category_separator(category,show_id)
pd.set_option('display.max_columns', None)



dframe_india.head(4)
# Total Number Comedy Movies present on NETFLIX



sns.countplot(x='Comedies',data=dframe_india,palette='Set2')

dframe_india['Comedies'].value_counts()
# Total Number Action & Adventure Movies present on NETFLIX



sns.countplot(x='Action & Adventure',data=dframe_india,palette='Set1')

dframe_india['Action & Adventure'].value_counts()
# Total Number Romantic Movies Movies present on NETFLIX



sns.countplot(x='Romantic Movies',data=dframe_india,palette='Set1')

dframe_india['Romantic Movies'].value_counts()
# Comedy Movies present in respective MOVIE TYPE categories



sns.countplot(y='Movie Type',hue='Comedies',data=dframe_india,palette='Set2')
# Romantic Movies present in respective MOVIE TYPE categories



sns.countplot(y='Movie Type',hue='Romantic Movies',data=dframe_india,palette='Set1')
# Horror Movies present in respective MOVIE TYPE categories



sns.countplot(y='Movie Type',hue='Horror Movies',data=dframe_india,palette='Set1')
# Comedy Bollywood Movies which we can see on NETFLIX



dframe_india['title'][dframe_india['Comedies']=='YES'].head(10)
# Action & Adventure Bollywood Movies which we can see on NETFLIX



dframe_india['title'][dframe_india['Action & Adventure']=='YES'].head(10)
# Romantic Movies Bollywood Movies which we can see on NETFLIX



dframe_india['title'][dframe_india['Romantic Movies']=='YES'].head(10)
# Adult Comedy Bollywood Movies present on NETFLIX



dframe_india['title'][(dframe_india['Comedies']=='YES')&(dframe_india['Movie Type']=='Adult Movies')]
# Family Comedy Bollywood Movies present on NETFLIX



dframe_india['title'][(dframe_india['Comedies']=='YES')&(dframe_india['Movie Type']=='Family Movies-Parental Guidance')].head(15)
# Horror Comedy Bollywood Movies present on NETFLIX



dframe_india['title'][(dframe_india['Comedies']=='YES')&(dframe_india['Horror Movies']=='YES')].head(10)
# Sports Bollywood Movies present on NETFLIX



dframe_india['title'][(dframe_india['Sports Movies']=='YES')].head(10)
# ACTION & ADVENTURE + DRAMA + FAMILY MOVIE present on NETFLIX



dframe_india['title'][(dframe_india['Action & Adventure']=='YES')&((dframe_india['Dramas']=='YES'))&(dframe_india['Movie Type']=='Family Movies-Parental Guidance')].head(15)
dframe_india.cast.isnull().sum()
# Filling up the NULL values in cast column



dframe_india['cast'].fillna(value='Actors Not Known',inplace=True)
dframe_india.cast.isnull().sum()
dframe_india.head()
# Creating a function for categorizing movies based on the actors



def actor_separator(actors,show_id):

    for a in (re.split(r',',actors)):

        if a.strip() in dframe_india:

            dframe_india[a.strip()][dframe_india['show_id']==show_id] = 'YES' 

        else:

            dframe_india[a.strip()]='NO'

            dframe_india[a.strip()][dframe_india['show_id']==show_id]='YES'
# Calling of function actor_separator for all the rows of the dataframe to categorize movies based on the actors



for show_id,actors in zip(dframe_india['show_id'],dframe_india['cast']):

    actor_separator(actors,show_id)
dframe_india.head(3)
# Salman Khan's movies available on NETFLIX



sns.countplot(y='Movie Type',hue='Salman Khan',data=dframe_india,palette='Set1')
# Total Number of Salman Khan movies present on NETFLIX



sns.countplot(x='Salman Khan',data=dframe_india,palette='Set1')

dframe_india['Salman Khan'].value_counts()
# Salman Khan all movies at NETFLIX



dframe_india['title'][dframe_india['Salman Khan']=='YES']
# Shah Rukh Khan's movies available on NETFLIX



sns.countplot(y='Movie Type',hue='Shah Rukh Khan',data=dframe_india,palette='Set1')
# Total Number of Shah Rukh Khan movies present on NETFLIX



sns.countplot(x='Shah Rukh Khan',data=dframe_india,palette='Set1')

dframe_india['Shah Rukh Khan'].value_counts()
# Shah Rukh Khan's ROMANTIC MOVIES



dframe_india['title'][(dframe_india['Shah Rukh Khan']=='YES')&(dframe_india['Romantic Movies']=='YES')]
# Shah Rukh Khan's COMEDY MOVIES



dframe_india['title'][(dframe_india['Shah Rukh Khan']=='YES')&(dframe_india['Comedies']=='YES')]
# Shah Rukh Khan's ACTION & ADVENTURE MOVIES



dframe_india['title'][(dframe_india['Shah Rukh Khan']=='YES')&(dframe_india['Action & Adventure']=='YES')]
# Shah Rukh Khan's ADULT Category movies



dframe_india['title'][(dframe_india['Shah Rukh Khan']=='YES')&(dframe_india['Movie Type']=='Adult Movies')]
# Akshay Kumar's movies available on NETFLIX



sns.countplot(y='Movie Type',hue='Akshay Kumar',data=dframe_india,palette='Set1')
# Total Number of Akshay Kumar movies present on NETFLIX



sns.countplot(x='Akshay Kumar',data=dframe_india,palette='Set1')

dframe_india['Akshay Kumar'].value_counts()
# Akshay Kumar's COMEDY MOVIES



dframe_india['title'][(dframe_india['Akshay Kumar']=='YES')&(dframe_india['Comedies']=='YES')]
# Akshay Kumar's ROMANTIC MOVIES



dframe_india['title'][(dframe_india['Akshay Kumar']=='YES')&(dframe_india['Romantic Movies']=='YES')]
# Akshay Kumar's ACTION & ADVENTURE MOVIES



dframe_india['title'][(dframe_india['Akshay Kumar']=='YES')&(dframe_india['Action & Adventure']=='YES')]
# Sunny Leone's ADULT CATEGORY MOVIES



dframe_india['title'][(dframe_india['Sunny Leone']=='YES')&(dframe_india['Movie Type']=='Adult Movies')]