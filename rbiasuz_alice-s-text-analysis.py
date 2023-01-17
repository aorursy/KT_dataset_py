import requests

import numpy as np # linear algebra

import pandas as pd # dataframes

from bs4 import BeautifulSoup # work with html

import nltk # natural language packages

import string # to do some work with strings

import matplotlib.pyplot as plt # data visualization

%matplotlib inline  

import seaborn as sns # data visualization

sns.set(color_codes=True) # data visualization

from textblob import TextBlob # sentiment analysis 
# Getting the Alice's HTML

try:

    r = requests.get('https://www.gutenberg.org/files/11/11-h/11-h.htm', verify=False)

except:

    print("Ops, not here")

    

# P.S.: Unhappy, we cannot run this code here in Kaggle, but I'll continue the process, and import the already extracted DataSet
tokens = []

with open("../input/tokens.txt", "r") as f:

    for line in f:

        tokens.append(str(line.strip()))

print(tokens[0:8])
# Looping through the tokens and make them lower case

words = []

for word in tokens:

    words.append(word.lower())

words[0:5]

# Removing the stopwords

from nltk.corpus import stopwords

#Here you may need download the stopwords: nltk.download('stopwords')

sw = stopwords.words('english')

words_ns = []

for word in words:

    if word not in sw:

        words_ns.append(word)

words_ns[0:5]

# Creating the word frequency distribution

freqdist = nltk.FreqDist(words_ns)

freqdist.most_common(30)
# What are doing gutenberg and project doing there? I also dislike some of these words:

new_stopwords = ['gutenberg', 'project', 'would', 'went', '1', 'e', 'tm', 'could', 'must']

word_final = []

for word in words_ns:

    if word not in new_stopwords:

        word_final.append(word)
# Plotting the word frequency distribution

freqdist = nltk.FreqDist(word_final)

plt.figure(figsize=(18,9))

plt.xlabel('Words', fontsize=18)

plt.ylabel('Freq', fontsize=16)

plt.xticks(size = 15)

plt.yticks(size = 15)

freqdist.plot(20)
# what about bigrams?

bigrams = list(nltk.bigrams(word_final))

freqdist = nltk.FreqDist(bigrams)

plt.figure(figsize=(18,9))

plt.xlabel('Words', fontsize=18)

plt.ylabel('Freq', fontsize=16)

plt.xticks(size = 15)

plt.yticks(size = 15)

freqdist.plot(25)

# what about trigrams?

trigrams = list(nltk.trigrams(word_final))

freqdist = nltk.FreqDist(trigrams)

freqdist.most_common(6)
#returning the text to a string format

all_text = "".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()

#here I'm at the start of the text, because the first charachters are descriptions of the site

all_text[2000:2500]

size = len(all_text)

all_text = all_text[2165:size]

size = len(all_text)

#Dividing in 30 parts

part_size = size/30

df = pd.DataFrame()

parts = []

for i in range(30):

    parts.append(all_text[int((i*part_size)):int((part_size*(i+1)))])
#turning into a dataframe (to easily manipulate

d = {'sentence':parts}

df = pd.DataFrame(d)
#The last five parts also are texts from the site, so we're droping it

df = df.drop(df.index[29])

df = df.drop(df.index[28])

df = df.drop(df.index[27])

df = df.drop(df.index[26])

df = df.drop(df.index[25])
#defining the diferent sentiment analysis methods:



#TextBlob:

def analize_sentiment_textblop(sentence):

    analysis = TextBlob(sentence)

    return analysis.sentiment.polarity



#Vader:

# analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_positive(sentence):

    score = analyser.polarity_scores(sentence)

    retorno = score.get('pos')

    return retorno



def sentiment_analyzer_negative(sentence):

    score = analyser.polarity_scores(sentence)

    retorno = score.get('neg')

    return retorno



def sentiment_analyzer_vader(sentence):

    score = analyser.polarity_scores(sentence)

    #print("{:-<40} {}".format(sentence, str(score)))

    retorno = score.get('compound')

    return retorno
df['textblop'] = df['sentence'].apply(lambda x: analize_sentiment_textblop(x))
df[['sentence','textblop']].plot(kind='bar', title='Sentiment / part of the book', figsize=(19,9), fontsize=10, colormap='viridis')

plt.show()