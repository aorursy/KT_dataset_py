import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS
gdf = pd.read_csv('../input/job_skills.csv')

gdf.shape
datadf = gdf[gdf['Title'].str.contains('Data Science') | gdf['Title'].str.contains('Data Scientist')]

datadf['Title'].count()

gdf[gdf['Title'].str.contains('Data Science') | gdf['Title'].str.contains('Data Scientist')]
mpl.rcParams['figure.figsize']=(8.0,7.0)

mpl.rcParams['font.size']=9                

mpl.rcParams['savefig.dpi']=100              

mpl.rcParams['figure.subplot.bottom']=.1 

#make the frame the wordcloud will be in



STOPWORDS.add('degree')

STOPWORDS.add('name')

STOPWORDS.add('dtype')

STOPWORDS.add('location')

stopwords = set(STOPWORDS)

data = datadf

#get stop words and the data



wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=100,

                          max_font_size=50, 

                          random_state=False

                         ).generate(str(data['Minimum Qualifications']))

#varibles for the wordcloud



print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

#show wordcloud
mpl.rcParams['figure.figsize']=(7.0,6.0)

mpl.rcParams['font.size']=9                

mpl.rcParams['savefig.dpi']=100              

mpl.rcParams['figure.subplot.bottom']=.1 

#make the frame the wordcloud will be in



stopwords = set(STOPWORDS)

data = datadf

#get stop words and the data



wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=100,

                          max_font_size=50, 

                          random_state=False

                         ).generate(str(data['Preferred Qualifications']))

#varibles for the wordcloud



print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

#show wordcloud
mpl.rcParams['figure.figsize']=(7.0,6.0)

mpl.rcParams['font.size']=9                

mpl.rcParams['savefig.dpi']=100              

mpl.rcParams['figure.subplot.bottom']=.1 

#make the frame the wordcloud will be in



STOPWORDS.add('degree')

stopwords = set(STOPWORDS)

data = datadf

#get stop words and the data



wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=100,

                          max_font_size=50, 

                          random_state=False

                         ).generate(str(data['Location']))

#varibles for the wordcloud



print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

#show wordcloud
mpl.rcParams['figure.figsize']=(7.0,6.0)

mpl.rcParams['font.size']=9                

mpl.rcParams['savefig.dpi']=100              

mpl.rcParams['figure.subplot.bottom']=.1 

#make the frame the wordcloud will be in



STOPWORDS.add('degree')

stopwords = set(STOPWORDS)

data = datadf

#get stop words and the data



wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=100,

                          max_font_size=50, 

                          random_state=False

                         ).generate(str(data['Responsibilities']))

#varibles for the wordcloud



print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

#show wordcloud