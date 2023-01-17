import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#for data visualization        

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
netflix = pd.read_csv('../input/netflix-shows/netflix_titles.csv')

netflix.head(5)
netflix.info()

#show_id and release_year are an integer and others are an object.

#from the dataset, we learn that column director, cast, country, date_added, and rating have different counts. We can check later, if there are NaN value or not.
netflix.isna().sum()

#there are NaN value, what we gonna do next is depends on what we need.

#we can erase the value, fill it with another value, or leave it like that if we not gonna use it.
#1. Groupby

summary_release = (netflix

                   .groupby('release_year')

                   .size().to_frame('show_id')

                   .reset_index())
#2. Change release_year type

summary_release['release_year'] = summary_release['release_year'].astype(str)
summary_release.dtypes
#3. Filter

summary_release = summary_release[(summary_release['show_id'] >= 30)]
#4. isualize it into bar type

plt.figure(figsize=(20,7))

graph = sns.barplot(data=summary_release, x='release_year', y='show_id')

plt.ylabel('Counts')

plt.xlabel('Release Year')

plt.title('Release Counts per Year')



for p in graph.patches:

        graph.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),

                    ha='center', va='bottom',

                    color= 'black')
#1. groupby

summary_rating = (netflix

                   .groupby('rating')

                   .size().to_frame('show_id')

                   .reset_index())
#2. We need additional step to make a better understanding about rating.

summary_rating
#we need to sort the rating, for viewer point of view, easier for them if we sort the data before.

summary_rating.sort_values(by=['show_id'], na_position='first', ascending=False, inplace=True)

summary_rating
#3. Visualize it into bar type 

plt.figure(figsize=(15,7))

graph = sns.barplot(data=summary_rating, x='rating', y='show_id')

plt.ylabel('Counts')

plt.xlabel('Rating')

plt.title('Counts per Rating')

plt.rcParams['font.size']=12 



for p in graph.patches:

        graph.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),

                    ha='center', va='bottom',

                    color= 'black')
#1. Groupby

summary_types = (netflix

                   .groupby('type')

                   .size().to_frame('show_id')

                   .reset_index()) 

summary_types
#2. Making a list of Movie and TV Show percentage

percent0 = round((summary_types['show_id'][0] / (summary_types['show_id'][0] + summary_types['show_id'][1])) * 100, 2)

percent1 = round((summary_types['show_id'][1] / (summary_types['show_id'][0] + summary_types['show_id'][1])) * 100, 2)

percent = [percent0, percent1]
#3. Visualize it into pie chart type

labels = ['Movie', 'TV Show']

sizes = [percent0, percent1]



fig1, ax = plt.subplots()

ax.pie(sizes, autopct='%1.1f%%', startangle=90)



ax.axis('equal')

plt.tight_layout()



plt.rcParams['font.sans-serif'] = 'Arial'

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['xtick.color'] = '#000000'

plt.rcParams['font.size']=20



ax.legend(labels=labels, frameon=False, bbox_to_anchor=(0.8,0.5))
#1. Making a list of actors

#2. Drop the NaN value

casts = [];

for cast in netflix.cast.dropna():

    casts.extend(str(cast).split(","))



casts2 = list(map(lambda x: x.strip(), casts))



import numpy as np

unique, counts = np.unique(casts2, return_counts=True)



cast_count = np.asarray((unique, counts)).T



summary_cast = pd.DataFrame(cast_count)
summary_cast.count()
summary_cast.dtypes
#3. Change summary_cast type

summary_cast[0] = summary_cast[0].astype(str)

summary_cast[1] = summary_cast[1].astype(int)
summary_cast.dtypes
#4. Filter

#5. Sort the summary_cast

summary_cast = summary_cast[(summary_cast[1] >= 20)]

summary_cast.sort_values(by=1, na_position='first', ascending=False, inplace=True)

summary_cast
#6. Visualize it into bar type

plt.figure(figsize=(24,8))

graph = sns.barplot(data=summary_cast, x=0, y=1)

plt.ylabel('Counts')

plt.xlabel('Actors')

plt.title('Actors Played in Movies or TV Show')

plt.rcParams['font.size']=10 



for p in graph.patches:

        graph.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),

                    ha='center', va='bottom',

                    color= 'black')
#1. Making a list of countries

#2. Drop the NaN value

countries = [];

for country in netflix.country.dropna():

    countries.extend(str(country).split(","))



countries2 = list(map(lambda x: x.strip(), countries))



import numpy as np

unique, counts = np.unique(countries2, return_counts=True)



cast_count = np.asarray((unique, counts)).T



summary_country = pd.DataFrame(cast_count)
summary_country.count()
summary_country.dtypes
#3. Change summary_country type

summary_country[0] = summary_country[0].astype(str)

summary_country[1] = summary_country[1].astype(int)
#4. Filter

#5. Sort the summary_cast

summary_country = summary_country[(summary_country[1] >= 100)]

summary_country.sort_values(by=1, na_position='first', ascending=False, inplace=True)
#6. Visualize it into bar type

plt.figure(figsize=(22.5,8))

graph = sns.barplot(data=summary_country, x=0, y=1)

plt.ylabel('Counts')

plt.xlabel('Countries')

plt.title('The Origin of Netflix Content')

plt.rcParams['font.size']=12 



for p in graph.patches:

        graph.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),

                    ha='center', va='bottom',

                    color= 'black')