# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from wordcloud import WordCloud, STOPWORDS 

import re

import nltk

from nltk.stem.porter import PorterStemmer

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/trump-tweets/trumptweets.csv')
dataset.tail()
dataset['date'] = pd.to_datetime(dataset['date'])
impeachment_start_date = '18-12-2019'

presedency_start_date = '17-01-2017'

start_election_campaign='16-6-2015'
dataset_during_impeachment_hearings=dataset.loc[dataset['date']>impeachment_start_date]

print('dataset during impeachment from 18-12-2019 to 20-1-2020') 

dataset_during_impeachment_hearings.count()

presedency_start_dataset=dataset.loc[dataset['date']>presedency_start_date]

presedency_before_dataset=dataset.loc[dataset['date']<presedency_start_date]

dataset_during_election_campaign=presedency_before_dataset.loc[presedency_before_dataset['date']>start_election_campaign]
corpus=[]

for i in range(40764,41121):

    review = re.sub('[^a-zA-Z]',' ',dataset_during_impeachment_hearings['content'][i])

    review=review.lower()

    review=review.split()

    ps=PorterStemmer()

#     review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    #review=' '.join(review)

    if 'https' in review:

        review.remove('https')

    if 'twitter' in review:

        review.remove('twitter')

        corpus=corpus+review

    else:

        corpus=corpus+review
stopwords = set(STOPWORDS)

listToStr = ' '.join([str(elem) for elem in corpus])

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(listToStr) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.title('During Impeachment')

plt.show()
corpus=[]

for i in range(31228,41121):

    review = re.sub('[^a-zA-Z]',' ',presedency_start_dataset['content'][i])

    review=review.lower()

    review=review.split()

    ps=PorterStemmer()

#     review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    #review=' '.join(review)

    if 'https' in review:

        review.remove('https')

    if 'twitter' in review:

        review.remove('twitter')

        corpus=corpus+review

    if 'pic' in review:

        review.remove('pic')

        corpus=corpus+review

    else:

        corpus=corpus+review
stopwords = set(STOPWORDS)

stopwords.add('pic')

listToStr = ' '.join([str(elem) for elem in corpus])

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(listToStr) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.title('During Whole Presidency')

plt.show()
corpus=[]

for i in range(0,31226):

    review = re.sub('[^a-zA-Z]',' ',presedency_before_dataset['content'][i])

    review=review.lower()

    review=review.split()

    ps=PorterStemmer()

#     review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    #review=' '.join(review)

    if 'https' in review:

        review.remove('https')

    if 'twitter' in review:

        review.remove('twitter')

        corpus=corpus+review

    if 'pic' in review:

        review.remove('pic')

        corpus=corpus+review

    else:

        corpus=corpus+review
stopwords = set(STOPWORDS)

stopwords.add('pic')

listToStr = ' '.join([str(elem) for elem in corpus])

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(listToStr) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.title('Before Presidency')

plt.tight_layout(pad = 0) 

plt.show()
corpus=[]

for i in range(23282,31227):

    review = re.sub('[^a-zA-Z]',' ',dataset_during_election_campaign['content'][i])

    review=review.lower()

    review=review.split()

    ps=PorterStemmer()

#     review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    #review=' '.join(review)

    if 'https' in review:

        review.remove('https')

    if 'twitter' in review:

        review.remove('twitter')

        corpus=corpus+review

    if 'pic' in review:

        review.remove('pic')

        corpus=corpus+review

    else:

        corpus=corpus+review
stopwords = set(STOPWORDS)

stopwords.add('pic')

listToStr = ' '.join([str(elem) for elem in corpus])

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(listToStr) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.title('During Election Campaign')

plt.tight_layout(pad = 0) 

plt.show()