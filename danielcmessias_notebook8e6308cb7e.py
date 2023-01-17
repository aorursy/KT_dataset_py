#Imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import math

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



#Characters

schars = pd.read_csv('../input/simpsons_characters.csv')

sepisodes = pd.read_csv('../input/simpsons_episodes.csv')

slines = pd.read_csv('../input/simpsons_script_lines.csv',error_bad_lines=False)
#Total characters

total_chars = len(schars)

#Total male

total_male = len(schars[schars.gender == 'm'])

#Total female

total_female = len(schars[schars.gender == 'f'])

#Total uncategorised

total_uncat = len(schars[pd.isnull(schars.gender)])



genderplot = schars.gender.value_counts().plot.pie(labels=('Male','Female'),figsize=(5,5),autopct='%1.1f%%')

genderplot.set_ylabel('')
fig = plt.figure()

ax = fig.add_subplot(111) # Create matplotlib axes

ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.



#Number of lines

clines = slines.groupby(['raw_character_text']).size().sort_values(ascending=False)[:10].plot(kind='bar',color='red',ax=ax)

clines.set_xlabel('Character')

clines.set_ylabel('Number of lines')



#Number of words



cwords = slines.groupby(['raw_character_text'])['word_count'].sum().sort_values(ascending=False)[:10].plot(kind='bar',color='blue',ax=ax2)

cwords.set_xlabel('Character')

cwords.set_ylabel('Number of words')
#Number of lines

locs = slines.groupby(['raw_location_text']).size().sort_values(ascending=False)[:10].plot(kind='bar',color='red')

locs.set_xlabel('Location')

locs.set_ylabel('Lines said at given location')
f,ax = plt.subplots(2,1,figsize=(10,5))

#Viewers

viewers = sepisodes.set_index('original_air_date')['us_viewers_in_millions']

viewers.index = pd.to_datetime(viewers.index)

viewers_plot = viewers.plot(ax=ax[0],x='test',color='m',figsize=(10,10))

viewers_plot.set_xlabel('')

viewers_plot.set_ylabel('Viewers (millions)')



#Ratings

rating = sepisodes.set_index('original_air_date')['imdb_rating']

rating.index = pd.to_datetime(rating.index)

rating_plot = rating.plot(ax=ax[1],x='test',color='c',figsize=(10,10))

rating_plot.set_xlabel('Original air date')

rating_plot.set_ylabel('IMDB Rating (out of 10)')
rating = sepisodes.set_index('original_air_date')['imdb_rating']

rating.index = pd.to_datetime(rating.index)

rating_plot = rating.plot(x='test',color='c',figsize=(10,10))

rating_plot.set_xlabel('Original air date')

rating_plot.set_ylabel('IMDB Rating (Out of 10)')