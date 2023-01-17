import os

import nltk

import nltk.corpus

import numpy as np, pandas as pd

from pandas import ExcelWriter

from pandas import ExcelFile

import matplotlib.pyplot as plt

import warnings

import itertools

from matplotlib.pyplot import figure

from PIL import Image

from os import path

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
df=pd.read_csv("../input/CORD-19-research-challenge/metadata.csv")
df.info()
print("There are {} observations and {} features in this dataset. \n".format(df.shape[0],df.shape[1]))
df.columns
df=pd.DataFrame(df)
df1=df['abstract'] # Create a subset of the dataset keeping only the abstract column
df2=df1.dropna() # Drop "NAs" from the dataset
df2=pd.DataFrame(df2)
df['word_count'] = df['abstract'].apply(lambda x: len(str(x).split(" ")))
print("There are {} rows and {} columns in the df2 dataset. \n".format(df2.shape[0],df2.shape[1]))
df2['abstract'] = df2['abstract'].str.replace('[^\w\s]','')

df2['abstract'].head()
from nltk.corpus import stopwords

stop = stopwords.words('english')

df2['abstract'] = df2['abstract'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

df2['abstract'].head()
# First identify the common words in the dataset
freq = pd.Series(' '.join(df2['abstract']).split()).value_counts()[:10]

freq
commonwords=("The", "In", "Abstract", "Our", "To", "Due", "This", "study", "BACKGROUND", "from", "a", "majority", 'However', 'time', 'specific', 

             'These', 'likely', 'known', 'studies', 'whose', 'well', 'per', 'identified', 'estimated', 'may', 'furthermore', 'poorly', 'changes',

             'remains', 'required', 'whose', "number", 'list', 'types', 'demonstrated', 'properties', 'found', 'addition', 'indepth', 'Therefore',

            'site', 'We', 'two', 'even', 'basis', 'found', 'including' 'also','certain', 'almost', 'using', 'All', 'Recent', 'followed', 'used',

             'one', 'several', 'filter', 'wasting', 'applicable', 'step', 'method', 'group', 'small', 'collected', 'developed', 'farm', 'Although', 'subsequent', 'list', 'can')
df2['abstract'] = df2['abstract'].apply(lambda x: " ".join(x for x in x.split() if x not in commonwords))

df2['abstract'].head()
rare = pd.Series(' '.join(df2['abstract']).split()).value_counts()[-10:]

rare
rare = list(rare.index)

df2['abstract'] = df2['abstract'].apply(lambda x: " ".join(x for x in x.split() if x not in rare))

df2['abstract'].head()
df2.columns
df2['abstract']=df2['abstract'].str.replace('\d+', '')
from textblob import Word

df2['abstract'] = df2['abstract'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

df2['abstract'].head()
df3= df2["abstract"].apply(nltk.word_tokenize)
df4=pd.DataFrame(df3)
df5=df4.to_string()
text=df4.abstract.values
wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',

    ).generate(str(text))
fig = plt.figure(

    figsize = (10, 8),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()