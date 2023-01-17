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



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv",encoding='latin-1')

data.head()
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "TweetText"]

data.columns = DATASET_COLUMNS

data.head()
data.drop(['ids','date','flag','user'],axis = 1,inplace = True)

data.head()
data['CleanTweet'] = data['TweetText'].str.replace("@", "") 

data.head()
data['CleanTweet'] = data['CleanTweet'].str.replace(r"http\S+", "") 

data.head()
data['CleanTweet'] = data['CleanTweet'].str.replace("[^a-zA-Z]", " ") 

data.head()
import nltk

stopwords=nltk.corpus.stopwords.words('english')



def remove_stopwords(text):

    clean_text=' '.join([word for word in text.split() if word not in stopwords])

    return clean_text



data['CleanTweet'] = data['CleanTweet'].apply(lambda text : remove_stopwords(text.lower()))

data.head()
data['CleanTweet'] = data['CleanTweet'].apply(lambda x: x.split())

data.head()
# from nltk.stem.porter import * 

# stemmer = PorterStemmer() 

# data['CleanTweet'] = data['CleanTweet'].apply(lambda x: [stemmer.stem(i) for i in x])

# data.head()
# data['CleanTweet'] = data['CleanTweet'].apply(lambda x: ' '.join([w for w in x]))

# data.head()
# data['CleanTweet'] = data['CleanTweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

# data.head()
# import matplotlib.pyplot as plt

# all_words = ' '.join([text for text in data['CleanTweet']])



# from wordcloud import WordCloud 

# wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words) 



# plt.figure(figsize=(10, 7)) 

# plt.imshow(wordcloud, interpolation="bilinear") 

# plt.axis('off') 

# plt.show()
# from textblob import TextBlob
# data['polarity'] = data['CleanTweet'].apply(lambda CleanTweet: TextBlob(CleanTweet).sentiment[0])

# data.head()
# data['subjectivity'] = data['CleanTweet'].apply(lambda CleanTweet: TextBlob(CleanTweet).sentiment[1])

# data.head()
# data['polarityinwords'] = np.where(data.polarity>0.000000000,'positive','negative')

# data.head()
# data['subjectivityinwords'] = np.where(data.subjectivity>0.5,'factual','personalopinion')



# data.head()
# data['polarityinwords'].value_counts()
# data['subjectivityinwords'].value_counts()
# positive_words =' '.join([text for text in data['CleanTweet'][data['polarityinwords'] =='positive']]) 

# wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(positive_words)



# plt.figure(figsize=(10, 7)) 

# plt.imshow(wordcloud, interpolation="bilinear") 

# plt.axis('off') 

# plt.show()
# depressive_words =' '.join([text for text in data['CleanTweet'][data['polarityinwords'] =='negative']]) 

# wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(depressive_words)



# plt.figure(figsize=(10, 7)) 

# plt.imshow(wordcloud, interpolation="bilinear") 

# plt.axis('off') 

# plt.show()