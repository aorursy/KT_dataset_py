# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import nltk
import pandas as pd
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.stem.porter import *
from wordcloud import WordCloud
from textblob import TextBlob
sact = pd.read_csv('../input/SATC_all_lines.csv')
sact.head()
sact.info()
sact['Line'].isnull
#carrie got most dialogue lines in all Seasons combined
sact['Speaker'].value_counts()[:15]
plt.figure(figsize=(15,10))
sact['Speaker'].value_counts()[:15].plot.bar()
plt.title('Sex and City Script of top 15 Characters')
plt.xticks(rotation=50)
plt.xlabel("Sex and City Characters")
plt.ylabel("Characters with max dialogues/script")
plt.show()
#created a new dataframe with 'Season','Episode','Speaker','Line'
new = sact[['Season','Episode','Speaker','Line']]
#considering carrie character lines/script of Entire Season 1
carrie = new[(new['Speaker']=='Carrie') & (new['Season']==1.0)]
carrie.head()

# Performs the following: on Line 
#     1. Remove all punctuation
#     2. Remove all stopwords
#     3. Remove stemming words
#     4. Returns a list of the cleaned text


def text_process(mess):
    stemmer = PorterStemmer()
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Remove stemming words(But didnt remove the stemming words , but if you want to remove the stemming words use code 
         [word for word in nopunc.split() if word.lower() not in stopwords.words('english') for word in nopunc.split() if word.lower() in stemmer.stem(word.lower())])
    4. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in (string.punctuation)]
    #join the no pucn words
    nopunc = ''.join(nopunc)

    #Removing the stopwords by converting the words to lower and check in stopwords and stemmer words
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    #return [stemmer.stem(token) for token in stop]
#[word for word in nopunc.split() if word.lower() not in stopwords.words('english') for word in nopunc.split() if word.lower() in stemmer.stem(word.lower())]
    

  
linesbySeason = carrie['Line'].size
linesbySeason

carrie['Line'][5]

text_process(carrie['Line'][5])
carrie_clean_lines = []


#Processing eac line to remove the punctuation,stopwords.
[carrie_clean_lines.append(text_process(carrie['Line'].iloc[i])) for i in range(0,linesbySeason)]
#Showing a sample of the carrie_clean_lines list
carrie_clean_lines[10]
len(carrie_clean_lines)
#Filtering the commong words
wordfreqdist = [nltk.FreqDist(carrie_clean_lines[i]).most_common(5) for i in range(0,len(carrie_clean_lines))]
print(len(wordfreqdist))

def text():
    """
     Here we are creating a new list- master with 5 most used words from each line of 1820 lines
    """
    masterlist = []
    for l in range(0,len(wordfreqdist)):
        #print(l)
        message = [''.join(word) for word, frequency in wordfreqdist[l]]
        #print(message)
        masterlist.extend(message)
        #print(masterlist)
    return' '.join(masterlist)
#Passing the masterlist text , which is converted to string

wordcloud = WordCloud(max_font_size=50, max_words=100,background_color='white').generate(text())
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis('off')
plt.show()
print(stopwords.words('english'))
def check_sentiment(samantha_line_sentiment):
    if samantha_line_sentiment['sentiment'] > 0.2:
        val = "Positive"
    elif samantha_line_sentiment['sentiment'] == 0:
        val = "Neutral"
    else:
        val = "Negative"
    return val
    
carrie_sentences = []

carrie_line_str =  carrie['Line'].astype(str)
for row in carrie_line_str:
    carrie_analysis = TextBlob(row)
    carrie_sentences.append((row, carrie_analysis.sentiment.polarity, carrie_analysis.sentiment.subjectivity))
    carrie_line_sentiment = pd.DataFrame(carrie_sentences, columns=['sentence', 'sentiment', 'polarity'])
    
def check_sentiment(carrie_line_sentiment):
    if carrie_line_sentiment['sentiment'] > 0.2:
        val = "Positive"
    elif carrie_line_sentiment['sentiment'] == 0:
        val = "Neutral"
    else:
        val = "Negative"
    return val

carrie_line_sentiment['Sentiment_Type'] = carrie_line_sentiment.apply(check_sentiment, axis=1)
plt.figure(figsize=(10, 10))
sns.set_style("whitegrid")
ax = sns.countplot(x="Sentiment_Type", data=carrie_line_sentiment)

plt.show()
#Samantha Lines/Dialogues Sentiment analysis
#considering Samantha character lines/script of Entire Season 1
samantha = new[(new['Speaker']=='Samantha') & (new['Season']==1.0)]


samantha_sentences = []

samantha_line_str =  samantha['Line'].astype(str)
for row in samantha_line_str:
    samantha_analysis = TextBlob(row)
    samantha_sentences.append((row, samantha_analysis.sentiment.polarity, samantha_analysis.sentiment.subjectivity))
    samantha_line_sentiment = pd.DataFrame(samantha_sentences, columns=['sentence', 'sentiment', 'polarity'])

def check_sentiment(samantha_line_sentiment):
    if samantha_line_sentiment['sentiment'] > 0.2:
        val = "Positive"
    elif samantha_line_sentiment['sentiment'] == 0:
        val = "Neutral"
    else:
        val = "Negative"
    return val
       
samantha_line_sentiment['Sentiment_Type'] = samantha_line_sentiment.apply(check_sentiment, axis=1)

plt.figure(figsize=(10, 10))
sns.set_style("whitegrid")
ax = sns.countplot(x="Sentiment_Type", data=samantha_line_sentiment)

plt.show()
#Miranda Lines/Dialogues Sentiment analysis
#considering Miranda character lines/script of Entire Season 1
miranda = new[(new['Speaker']=='Miranda') & (new['Season']==1.0)]

miranda_sentences = []

miranda_line_str =  miranda['Line'].astype(str)
for row in miranda_line_str:
    miranda_analysis = TextBlob(row)
    miranda_sentences.append((row, miranda_analysis.sentiment.polarity, miranda_analysis.sentiment.subjectivity))
    miranda_line_sentiment = pd.DataFrame(miranda_sentences, columns=['sentence', 'sentiment', 'polarity'])

def check_sentiment(miranda_line_sentiment):
    if miranda_line_sentiment['sentiment'] > 0.2:
        val = "Positive"
    elif miranda_line_sentiment['sentiment'] == 0:
        val = "Neutral"
    else:
        val = "Negative"
    return val
       
miranda_line_sentiment['Sentiment_Type'] = miranda_line_sentiment.apply(check_sentiment, axis=1)

plt.figure(figsize=(10, 10))
sns.set_style("whitegrid")
ax = sns.countplot(x="Sentiment_Type", data=miranda_line_sentiment)

plt.show()
#Charlotte Lines/Dialogues Sentiment analysis
#considering Charlotte character lines/script of Entire Season 1
charlotte = new[(new['Speaker']=='Charlotte') & (new['Season']==1.0)]
charlotte_sentences = []

charlotte_line_str =  charlotte['Line'].astype(str)
for row in charlotte_line_str:
    charlotte_analysis = TextBlob(row)
    charlotte_sentences.append((row, charlotte_analysis.sentiment.polarity, charlotte_analysis.sentiment.subjectivity))
    charlotte_line_sentiment = pd.DataFrame(charlotte_sentences, columns=['sentence', 'sentiment', 'polarity'])

def check_sentiment(charlotte_line_sentiment):
    if charlotte_line_sentiment['sentiment'] > 0.2:
        val = "Positive"
    elif charlotte_line_sentiment['sentiment'] == 0:
        val = "Neutral"
    else:
        val = "Negative"
    return val
       
charlotte_line_sentiment['Sentiment_Type'] = charlotte_line_sentiment.apply(check_sentiment, axis=1)

plt.figure(figsize=(10, 10))
sns.set_style("whitegrid")
ax = sns.countplot(x="Sentiment_Type", data=charlotte_line_sentiment)

plt.show()