import numpy as np

import pandas as pd
education = pd.read_csv("../input/india-national-education-policy2020-tweets-dataset/NEP_2020_english_tweet.csv")
education.head()
from textblob import TextBlob

import nltk

from nltk.corpus import stopwords

import string
stopword = stopwords.words('english')
stopword = [word for word in stopword if "n't" not in word] #I am not removing words like hasn't,haven't etc. 
stopword
def preprocess(texts):

    text = texts.strip()

    text_word = [word.lower() for word in text.split() if "#" not in word]

    text = " ".join(text_word)

    text_word = [word for word in text_word if "@" not in word]

    text = " ".join(text_word)

    text_word = [word for word in text_word if "https:" not in word]

    text = " ".join(text_word)

    text_word = [word for word in text_word if "http:" not in word]

    text = " ".join(text_word)

    text_word = [word for word in text_word if "www." not in word]

    text = " ".join(text_word)

    text_ch = [ch if ch not in string.punctuation else " " for ch in text]

    text = "".join(text_ch)

    text_word = [word.strip() for word in text.split() if word.strip() not in stopword]

    text = " ".join(text_word)

    text_word = [word for word in text_word if len(word) > 1]

    text = " ".join(text_word)

    text_word = [word.strip() for word in text.split() if not word.isdigit()] 

    return " ".join(text_word)
education['Processed'] = education.Tweet.apply(preprocess)
def getSubjectivity(text):

    return TextBlob(text).sentiment.subjectivity



#Creating a function to get polarity

def getpolarity(text):

    return TextBlob(text).sentiment.polarity



#Create two new columns

education['subjectivity']=education['Processed'].apply(getSubjectivity) #Subjective sentences generally refer to personal opinion, emotion or judgment rather than factual information.

education['polarity']=education['Processed'].apply(getpolarity) #it tell us actual sentiment whether positive negative or neutral.

                                                                #polarity > 0 means positive sentiment

                                                                #polarity 0 means neutral while < 0 means negative sentiment.
education.head()
#Converting float polarity into categorical ones.

def getAnalysis(score):

    if score<0:

        return 'Negative'

    elif score == 0 :

        return 'Neutral'

    else:

        return 'Positive'
education['Sentiment'] = education.polarity.apply(getAnalysis)
senti_count = education.Sentiment.value_counts()
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))

senti_count.plot.bar()

plt.xlabel("Sentiments")

plt.ylabel("Frequency")

plt.show()
plt.pie(x=senti_count.values,labels=senti_count.index,shadow=True,explode=(0,0.1,0.1),autopct='%1.1f%%',radius=1.5)

plt.show()