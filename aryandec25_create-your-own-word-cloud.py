# for some basic operations

import numpy as np

import pandas as pd



# for visualizations

import matplotlib.pyplot as plt

import seaborn as sns



# for providing path

import os

print(os.listdir('../input/final-wordcloud/'))
# reading the data



data = pd.read_excel('../input/final-wordcloud/final_wordcloud.xlsx')



# getting the shape of the data

data.shape
data.head()
# Null values in message column

data['message'].isnull().sum()
# Dropping all null values from DataFrame

data.dropna(inplace=True)
data.isnull().sum()
data['postedBy'].unique()
flipkart=data[data['postedBy']=='Flipkart']

Amazon=data[data['postedBy']=='Amazon India']

Snapdeal=data[data['postedBy']=='Snapdeal']

Myntra=data[data['postedBy']=='Myntra']

BTP=data[data['postedBy']=='Bengaluru Traffic Police']

HTP=data[data['postedBy']=='Hyderabad Traffic Police']

KTP=data[data['postedBy']=='Kolkata Traffic Police']

Idea=data[data['postedBy']=='Idea']

Tatadocomo=data[data['postedBy']=='Tata Docomo']

Aircel=data[data['postedBy']=='Aircel India']

Fortis=data[data['postedBy']=='Fortis Healthcare']

Ambani=data[data['postedBy']=='Kokilaben Dhirubhai Ambani Hospital']

Apollo=data[data['postedBy']=='Apollo Hospitals']

Modi=data[data['postedBy']=='Narendra Modi']

Rahul=data[data['postedBy']=='Rahul Gandhi']

Kejriwal=data[data['postedBy']=='Arvind Kejriwal']
#Importing word cloud and Stopwords

from wordcloud import WordCloud

from wordcloud import STOPWORDS
stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'red',

                      width = 2000,

                      height = 2000).generate(str(flipkart['message']))



plt.rcParams['figure.figsize'] = (12, 12)

plt.axis('off')

plt.imshow(wordcloud)

plt.title('Word Cloud: Flipkart', fontsize = 30)

plt.show()
stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'green',

                      width = 2000,

                      height = 2000).generate(str(Amazon['message']))



plt.rcParams['figure.figsize'] = (12, 12)

plt.axis('off')

plt.imshow(wordcloud)

plt.title('Word Cloud: Amazon', fontsize = 30)

plt.show()
stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'grey',

                      width = 2000,

                      height = 2000).generate(str(Snapdeal['message']))



plt.rcParams['figure.figsize'] = (12, 12)

plt.axis('off')

plt.imshow(wordcloud)

plt.title('Word Cloud: Snapdeal', fontsize = 30)

plt.show()
stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'cyan',

                      width = 2000,

                      height = 2000).generate(str(Myntra['message']))



plt.rcParams['figure.figsize'] = (12, 12)

plt.axis('off')

plt.imshow(wordcloud)

plt.title('Word Cloud: Myntra', fontsize = 30)

plt.show()
stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'yellow',

                      width = 2000,

                      height = 2000).generate(str(BTP['message']))



plt.rcParams['figure.figsize'] = (12, 12)

plt.axis('off')

plt.imshow(wordcloud)

plt.title('Word Cloud: Bengaluru Traffic Police', fontsize = 30)

plt.show()
stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'brown',

                      width = 2000,

                      height = 2000).generate(str(HTP['message']))



plt.rcParams['figure.figsize'] = (12, 12)

plt.axis('off')

plt.imshow(wordcloud)

plt.title('Word Cloud: Hyderabad Traffic Police', fontsize = 30)

plt.show()
stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'grey',

                      width = 2000,

                      height = 2000).generate(str(KTP['message']))



plt.rcParams['figure.figsize'] = (12, 12)

plt.axis('off')

plt.imshow(wordcloud)

plt.title('Word Cloud: Kolkata Traffic Police', fontsize = 30)

plt.show()
stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'yellow',

                      width = 2000,

                      height = 2000).generate(str(Idea['message']))



plt.rcParams['figure.figsize'] = (12, 12)

plt.axis('off')

plt.imshow(wordcloud)

plt.title('Word Cloud: Idea', fontsize = 30)

plt.show()
stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'cyan',

                      width = 2000,

                      height = 2000).generate(str(Tatadocomo['message']))



plt.rcParams['figure.figsize'] = (12, 12)

plt.axis('off')

plt.imshow(wordcloud)

plt.title('Word Cloud: Tata Docomo', fontsize = 30)

plt.show()
stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'yellow',

                      width = 2000,

                      height = 2000).generate(str(Aircel['message']))



plt.rcParams['figure.figsize'] = (12, 12)

plt.axis('off')

plt.imshow(wordcloud)

plt.title('Word Cloud: Aircel India', fontsize = 30)

plt.show()
stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'yellow',

                      width = 2000,

                      height = 2000).generate(str(Fortis['message']))



plt.rcParams['figure.figsize'] = (12, 12)

plt.axis('off')

plt.imshow(wordcloud)

plt.title('Word Cloud: Fortis Healthcare', fontsize = 30)

plt.show()
stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'blue',

                      width = 2000,

                      height = 2000).generate(str(Ambani['message']))



plt.rcParams['figure.figsize'] = (12, 12)

plt.axis('off')

plt.imshow(wordcloud)

plt.title('Word Cloud: Kokilaben Dhirubhai Ambani Hospital', fontsize = 30)

plt.show()
stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'brown',

                      width = 2000,

                      height = 2000).generate(str(Apollo['message']))



plt.rcParams['figure.figsize'] = (12, 12)

plt.axis('off')

plt.imshow(wordcloud)

plt.title('Word Cloud: Apollo Hospiltal', fontsize = 30)

plt.show()
stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'yellow',

                      width = 2000,

                      height = 2000).generate(str(Modi['message']))



plt.rcParams['figure.figsize'] = (12, 12)

plt.axis('off')

plt.imshow(wordcloud)

plt.title('Word Cloud: Narendra Modi', fontsize = 30)

plt.show()
stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'green',

                      width = 2000,

                      height = 2000).generate(str(Rahul['message']))



plt.rcParams['figure.figsize'] = (12, 12)

plt.axis('off')

plt.imshow(wordcloud)

plt.title('Word Cloud: Rahul Gandhi', fontsize = 30)

plt.show()
stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'red',

                      width = 2000,

                      height = 2000).generate(str(Kejriwal['message']))



plt.rcParams['figure.figsize'] = (12, 12)

plt.axis('off')

plt.imshow(wordcloud)

plt.title('Word Cloud: Arvind Kejriwal', fontsize = 30)

plt.show()