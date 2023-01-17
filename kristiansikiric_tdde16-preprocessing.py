# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv("/kaggle/input/lyrics-dataset/songs_dataset.csv")

df = df[["Lyrics","Genre"]]

# Any results you write to the current directory are saved as output.
import spacy

nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser"])



def preprocess_genre(df):

    n = len(df)

    for i in range(n):

        genre = df["Genre"].iloc[i]

        #print(genre)

        genre = genre.replace('[','')

        genre = genre.replace(']','')

        genre = genre.replace("'",'')

        genre = genre.split(', ')

        if not len(genre) == 1:

            new_genre = ""

            tmp_count = 0

            count = 0

            for g in genre:

                try:

                    count = df.Genre.value_counts()["['"+g+"']"]

                except:

                    next

                if  count > tmp_count:

                    #print(g)

                    new_genre = g

                    tmp_count = count

            df["Genre"].iloc[i] = new_genre

        else:

            #print(genre[0])

            df["Genre"].iloc[i] = genre[0]

    return df



def clean_lyrics(text):

    text = text.lower()

    text = text.replace('verse','')

    text = text.replace('hook','')

    text = text.replace('chorus','')

    text = text.replace('intro','')

    text = nlp(text)

    text = [t.text for t in text if t.is_alpha and not t.is_stop]

    text = ' '.join(text)

    return(text)



df_processed = preprocess_genre(df)#.sample(frac = 0.005, random_state = 123))

print(df_processed["Genre"].value_counts())

df_processed["Genre"].value_counts().plot(kind='bar')



df_processed["Lyrics"] = df_processed["Lyrics"].apply(clean_lyrics)

print(df_processed.head())



df_processed.to_csv("processed_lyrics_genres.csv", index = False)