# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3 # read the database

from nltk.collocations import *

import nltk

from sklearn.decomposition import PCA # for principal componenet analysis

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# connect to database

conn = sqlite3.connect('../input/database.sqlite')

#extract data from database into a dataframe

df = pd.read_sql_query("SELECT * FROM BoardGames", conn)



# verify that result of SQL query is stored in the dataframe

print(df.head())

print(df.shape)

conn.close()
for col in df.columns:

    print("*",col,"-","datatype:",df[col].dtype,", unique values:",len(df[col].unique()))
df['game.type'].unique()
print("there are","{:,}".format(df[df['game.type']=='boardgame'].shape[0]),"boardgames")
df = df[df['game.type']=='boardgame']
for col in df.columns:

    if (len(df[col].unique()) == 76688) & (('details' in col) | ('attributes' in col)):

        print("*",col,"-","datatype:",df[col].dtype,", all unique")

    elif ('details' in col) | ('attributes' in col):

        print("**",col,"-","datatype:",df[col].dtype,", unique values:",len(df[col].unique()))
df['polls.suggested_playerage'] = df['polls.suggested_playerage'].fillna(0).astype('int64')
# This is to grab columns by type. 

# number of columns in dataframe

s = df.shape[1]

# emtpy list to store columns that will be kept

text_columns = []

number_columns = []

other_columns = []





c = list(df.columns)



for i in range(s):

    if df.iloc[:,i].dtype == 'object':

        text_columns.append(c[i])

    elif (df.iloc[:,i].dtype == 'float64') or (df.iloc[:,i].dtype == 'int64'):

        number_columns.append(c[i])

    else:

        other_columns.append(c[i])



print("Text Columns",len(text_columns))

print("Number Columns",len(number_columns))

print("Other Columns",len(other_columns))

text_columns.remove('details.image')

text_columns.remove('details.thumbnail')
for col in text_columns:

    print(col)
len(df['game.id']) == len(df['game.id'].unique())
len(df['row_names']) == len(df['row_names'].unique())
(df['game.id'] == df['row_names']).unique()
text_df = df[text_columns].set_index(['row_names', 'game.id'])
text_df['attributes.boardgameartist'] = text_df['attributes.boardgameartist'].str.split(',')

text_df['attributes.boardgamedesigner'] = text_df['attributes.boardgamedesigner'].str.split(',')
def my_tokens(row):

    try:

        word_list = nltk.word_tokenize(row)

    except:

        word_list = None

    return word_list
text_columns.remove('attributes.t.links.concat.2....')

text_columns.remove('attributes.boardgameartist')

text_columns.remove('attributes.boardgamedesigner')

text_columns.remove('polls.language_dependence')
suggested_options = set([])

for col in text_columns:

    if 'polls.suggested_numplayers' in col:

        suggested_options = suggested_options | set(text_df[col].unique())

    else:

        try:

            text_df[col] = text_df[col].apply(my_tokens)

        except:

            print(col)
text_df
text_df.columns
get_words_from = ['details.description', 'details.name',

       'attributes.boardgameartist', 'attributes.boardgamecategory',

       'attributes.boardgamecompilation', 'attributes.boardgamedesigner',

       'attributes.boardgameexpansion', 'attributes.boardgamefamily',

       'attributes.boardgameimplementation', 'attributes.boardgameintegration',

       'attributes.boardgamemechanic', 'attributes.boardgamepublisher',

       'attributes.t.links.concat.2....']

words = {}

for col in get_words_from:

    working_list = list(text_df[col])

    for i in working_list:

        if i != None:

            for j in i:

                j = stemmer.stem(j)

                if j in words:

                    words[j] += 1

                else:

                    words[j] = 1

len(words)
words