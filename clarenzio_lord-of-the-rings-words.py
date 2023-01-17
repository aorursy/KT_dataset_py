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
# we will focus on df_words in the analysis by applying multi-level indexing

# import json



# with open('/kaggle/input/lord-of-the-rings-character-data/LordOfTheRingsBook.json', 'r') as f:

#     data = f.read()

# data_book = json.loads(data)



# df_characters = pd.read_csv('/kaggle/input/lord-of-the-rings-character-data/Characters.csv')

# df_movies = pd.read_csv('/kaggle/input/lord-of-the-rings-character-data/Movies.csv')

df_words = pd.read_csv('/kaggle/input/lord-of-the-rings-character-data/WordsByCharacter.csv')
df_words.info()
df_words.head(10)
df_cat_describe = df_words.select_dtypes(include='O').describe()

df_cat_describe
df_top_character = df_words.loc[df_words.Character==df_cat_describe.loc['top','Character'],:]

print(len(df_top_character))

print(df_top_character.Film.unique())
# default index

print(df_words.index)

print(type(df_words.index.values)) # return the underlying data in some unique data type by .values attribute

print(df_words.index.name) # by default, the index column of a df in RangeIndex, whose name attribute is None
# apply multi-indexing, use all categorical columns, and sort index to ensure efficiency

df_words_multi = df_words.set_index(['Film','Chapter','Race','Character']).sort_index()

df_words_multi.head(10)
# check for index after applying multi-indexing

print(df_words_multi.index.names)

print(df_words_multi.index.values[:10])

print('size of original data:',len(df_words))

print('size of multi-indexed data:',len(df_words_multi))
# apply reset_index to restore any level of multi-index

df_new = df_words_multi.reset_index(level=['Chapter','Race'])

df_original = df_words_multi.reset_index() # by default, reset_index removes all levels of indices and add to columns

df_new.head(10)
# Q1: Which characters speak in the first chapter of “The Fellowship of the Ring”?



# method 1: using single indexing + boolean operator

print(df_words.loc[(df_words.Film=='The Fellowship Of The Ring')&(df_words.Chapter=='01: Prologue'),'Character'].unique())



# method 2: using multi-indexing (add sort_index() to boost performance, but this takes extra memory)

print(df_words.set_index(['Film','Chapter']).sort_index().loc[('The Fellowship Of The Ring','01: Prologue'),'Character'].unique())



# method 2(concise): the multi-index does not need to be of same length between query and df

print(df_words_multi.loc[('The Fellowship Of The Ring','01: Prologue'),:].reset_index(level='Race',drop=True).index.unique().values)
# Q2: Who are the first three elves to speak in the “The Fellowship of the Ring”? 



# method 1

idx = pd.IndexSlice

print(df_words_multi.loc[idx['The Fellowship Of The Ring',:,'Elf',:]].reset_index(level='Character').iloc[0:3,:]['Character'].values)



# method 2

print(df_words_multi.loc[('The Fellowship Of The Ring',slice(None),'Elf'),:].head(3).reset_index(level='Character')['Character'].values)



# .head() is a nice shorthand for .iloc() 
# Q3: How much do Gandalf and Saruman talk in each chapter of “The Two Towers”? (indexing multiple possible values)

        

# method 1

df_words_multi.loc[('The Two Towers',slice(None),slice(None),['Gandalf','Saruman']),:]
# Q4: How much does Isildur speak in all of the films? (use pd.DataFrame.xs to get cross-section)



# method 1:

df_grouped = df_words.groupby(['Film','Chapter','Race','Character']).sum() 

df_grouped.loc[(slice(None),slice(None),slice(None),'Galadriel'),:]
# method 2:

df_words_multi.xs('Galadriel',level='Character')
# Q5: Which hobbits speak the most in each film and across all three films? (use pivot table to both display single record & aggregated record)



# method 1:

# in each film

df_hobbit_each_film = df_words.loc[df_words.Race=='Hobbit'].groupby(['Film','Character']).sum().sort_values(['Film','Words'],ascending=False)

for film in df_words.Film.unique():

    print('Hobbits speaking the most in {}'.format(film))

#     print(df_hobbit_each_film.xs(film,level='Film').head(3))

    print(df_hobbit_each_film.loc[film,:].head(3))



# across all three films

print('Hobbits speaking the most across all 3 films:')

print(df_words.loc[df_words.Race=='Hobbit'].groupby(['Race','Character']).sum().sort_values(['Words'],ascending=False).head(3))
# method 2:

pivoted = df_words.pivot_table(index = ['Race','Character'],

                               columns = 'Film',

                               aggfunc = 'sum',

                               margins = True,

                               margins_name = 'All Films',

                               fill_value = 0).sort_index()



pivoted = pivoted.sort_values(by=('Words','All Films'),ascending=False) # notice the index is not aggregated!

pivoted
pivoted.loc['Hobbit']
# How much does each race speak in different films

pivot_film_race = df_words.pivot_table(index=['Film','Chapter'],columns='Race',values='Words',aggfunc='sum',fill_value=0)

# pivot_film_race

pivot_film_race.loc[('The Fellowship Of The Ring',slice(None))]
# to be continued...