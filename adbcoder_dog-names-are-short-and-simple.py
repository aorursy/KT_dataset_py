# imports

%matplotlib inline

import warnings

warnings.filterwarnings("ignore", message="axes.color_cycle is deprecated")

import numpy as np

import pandas as pd

import scipy as sp

import seaborn as sns

import matplotlib.pyplot as plt

import wordcloud
df = pd.read_csv('../input/dogNames2.csv', names=['name', 'count'], header=0)

df.name = df.name.apply(lambda x: str(x))

df.head()
# Lets take a look at the overall statistics.

df.describe()
df.sort_values(by='count', ascending=False).head(10).plot.bar(x='name', y='count')
# A random sample of unpopular names

df.query('count <= 10').sample(10, random_state=5)
df['name_length'] = df.name.apply(lambda x: len(str(x)))
# visualize name poplularity vs length correlation:

df[['name_length', 'count']].plot.scatter(x='name_length', y='count')
def CountSyllables(word, isName=True):

    vowels = "aeiouy"

    #single syllables in words like bread and lead, but split in names like Breanne and Adreann

    specials = ["ia","ea"] if isName else ["ia"]

    specials_except_end = ["ie","ya","es","ed"]  #seperate syllables unless ending the word

    currentWord = word.lower()

    numVowels = 0

    lastWasVowel = False

    last_letter = ""



    for letter in currentWord:

        if letter in vowels:

            #don't count diphthongs unless special cases

            combo = last_letter+letter

            if lastWasVowel and combo not in specials and combo not in specials_except_end:

                lastWasVowel = True

            else:

                numVowels += 1

                lastWasVowel = True

        else:

            lastWasVowel = False



        last_letter = letter



    #remove es & ed which are usually silent

    if len(currentWord) > 2 and currentWord[-2:] in specials_except_end:

        numVowels -= 1



    #remove silent single e, but not ee since it counted it before and we should be correct

    elif len(currentWord) > 2 and currentWord[-1:] == "e" and currentWord[-2:] != "ee":

        numVowels -= 1



    return numVowels
df['syllables'] = df.name.apply(lambda x: CountSyllables(str(x)))
# visualize name poplularity vs num if syllables:

df[['syllables', 'count']].plot.scatter(x='syllables', y='count')
allsongs = ' '.join(df.text).lower().replace('choru', '')

cloud = wordcloud.WordCloud(background_color='white',

                            max_font_size=100,

                            width=1000,

                            height=500,

                            max_words=300,

                            relative_scaling=.5).generate(allsongs)

plt.figure(figsize=(10,5))

plt.axis('off')

plt.savefig('allsongs.png')

plt.imshow(cloud);
cloud = wordcloud.WordCloud(background_color='white',

                            max_font_size=150,

                            width=1000,

                            height=500,

                            max_words=df.size,

                            relative_scaling=.5).generate_from_frequencies(df.set_index('name')['count'].to_dict())

plt.figure(figsize=(10,5))

plt.axis('off')

plt.imshow(cloud);