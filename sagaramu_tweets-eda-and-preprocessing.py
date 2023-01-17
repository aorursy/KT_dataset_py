#POS tagging code

'''CC coordinating conjunction

CD cardinal digit

DT determiner

EX existential there (like: "there is" ... think of it like "there exists")

FW foreign word

IN preposition/subordinating conjunction

JJ adjective 'big'

JJR adjective, comparative 'bigger'

JJS adjective, superlative 'biggest'

LS list marker 1)

MD modal could, will

NN noun, singular 'desk'

NNS noun plural 'desks'

NNP proper noun, singular 'Harrison'

NNPS proper noun, plural 'Americans'

PDT predeterminer 'all the kids'

POS possessive ending parent's

PRP personal pronoun I, he, she

PRP$ possessive pronoun my, his, hers

RB adverb very, silently,

RBR adverb, comparative better

RBS adverb, superlative best

RP particle give up

TO to go 'to' the store.

UH interjection errrrrrrrm

VB verb, base form take

VBD verb, past tense took

VBG verb, gerund/present participle taking

VBN verb, past participle taken

VBP verb, sing. present, non-3d take

VBZ verb, 3rd person sing. present takes

WDT wh-determiner which

WP wh-pronoun who, what

WP$ possessive wh-pronoun whose

WRB wh-abverb where, when'''



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
# General Libraries

import nltk

import matplotlib.pyplot as plt

import seaborn as sns

import re

# specific for data preproressing and visualization

from nltk.tokenize import sent_tokenize,word_tokenize

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer

from wordcloud import WordCloud,STOPWORDS 

from statistics import mode

# classifiers

from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC,LinearSVC,NuSVC

from nltk.classify import ClassifierI
TweetData =  pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
TweetData.head()
TweetData.info()
print(TweetData.isna().any())

TweetData[TweetData.isnull().any(axis=1)]
TweetData.dropna(inplace=True)
TweetData.info()
# the count of sentiments in absolute value as we as percentage form



plt.figure(figsize=(12,6))

plt.subplot(1,2,1)

sns.countplot(TweetData['sentiment'], facecolor=(0, 0, 0, 0),

                   linewidth=5,

                   edgecolor=sns.color_palette("muted", 3))

plt.subplot(1,2,2)

(TweetData['sentiment'].value_counts()/len(TweetData)*100).plot(kind="bar", rot=0)
# lets count the words in each tweet and add columns for it

def wordcount(text):

    words = word_tokenize(text)

    return len(words)



TweetData['Text_words'] = TweetData['text'].apply(lambda x : wordcount(x))

TweetData['Select_text_words']= TweetData['selected_text'].apply(lambda x : wordcount(x))
TweetData.head()
sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})

sns.set_context("paper", font_scale=2)

plt.figure(figsize=(12,6),dpi=100)

plt.subplot(1,2,1)

sns.distplot(TweetData['Text_words'],kde=False)

plt.subplot(1,2,2)

sns.distplot(TweetData['Select_text_words'],kde=False)
stop_words = set(stopwords.words('english'))

def clean_tweets(x):

    # removing the hyperlinks

    clean1 = re.sub('https?://[A-Za-z0-9./]+','',x)

    # removing the hashtags

    clean2 = re.sub('#[A-Za-z0-9]+','',clean1)

    # removing @

    clean3 = re.sub('@[A-Za-z0-9]','',clean2)

    # removing punctuations and lower case conversion

    clean4 = re.sub(r'[^\w\s]','',clean3).lower()

    words = word_tokenize(clean4)

    # removing stopwords

    words = [w for w in words if not w in stop_words]

    sent = ' '.join(words)

    return sent
# lets first check it for few sentences:

for i in range(12,30):

    Clean = clean_tweets(TweetData['text'][i])

    print("Text: ",TweetData['text'][i])

    print("Clean Text: ", Clean)
# lets apply our clean tweet funcion for all tests

TweetData['Clean_tweet'] = TweetData['text'].apply(lambda x : clean_tweets(x))

TweetData['Clean_tweet_words'] = TweetData['Clean_tweet'].apply(lambda x : wordcount(x))
TweetData.head()
sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})

sns.set_context("paper", font_scale=1.2)

plt.figure(figsize=(12,6),dpi=100)

plt.subplot(1,3,1)

sns.distplot(TweetData['Text_words'],kde=False)

plt.subplot(1,3,2)

sns.distplot(TweetData['Select_text_words'],kde=False)

plt.subplot(1,3,3)

sns.distplot(TweetData['Clean_tweet_words'],kde=False)
# All word Freqency curve with out any pos tagging filter

All_words = []

for sent in TweetData['Clean_tweet']:

    for word in word_tokenize(sent):

            All_words.append(word)

All_words_freq = nltk.FreqDist(All_words)

Freq_word_DF = pd.DataFrame({"Data":All_words_freq.most_common(15)})

Freq_word_DF['Words'] = Freq_word_DF['Data'].apply(lambda x : x[0])

Freq_word_DF['freq'] = Freq_word_DF['Data'].apply(lambda x : x[1])

sns.set()

sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})

sns.set_context("paper", font_scale=2)

fig=plt.figure(figsize =(20,8),dpi=50)

sns.barplot('Words','freq',data = Freq_word_DF)
# TweetData[TweetData['sentiment']=='positive']['Clean_tweet']

All_words = []

for sent in TweetData[TweetData['sentiment']=='positive']['Clean_tweet']:

    for word in word_tokenize(sent):

            All_words.append(word)

All_words_freq = nltk.FreqDist(All_words)

Freq_word_DF = pd.DataFrame({"Data":All_words_freq.most_common(15)})

Freq_word_DF['Words'] = Freq_word_DF['Data'].apply(lambda x : x[0])

Freq_word_DF['freq'] = Freq_word_DF['Data'].apply(lambda x : x[1])

sns.set()

sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})

sns.set_context("paper", font_scale=2)

fig=plt.figure(figsize =(20,8),dpi=50)

ax = sns.barplot('Words','freq',data = Freq_word_DF)

ax.set_title("Positive Tweets Words Frequency Plot")



# Negative tweets

All_words = []

for sent in TweetData[TweetData['sentiment']=='negative']['Clean_tweet']:

    for word in word_tokenize(sent):

            All_words.append(word)

All_words_freq = nltk.FreqDist(All_words)

Freq_word_DF = pd.DataFrame({"Data":All_words_freq.most_common(15)})

Freq_word_DF['Words'] = Freq_word_DF['Data'].apply(lambda x : x[0])

Freq_word_DF['freq'] = Freq_word_DF['Data'].apply(lambda x : x[1])

sns.set()

sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})

sns.set_context("paper", font_scale=2)

fig=plt.figure(figsize =(20,8),dpi=50)

ax = sns.barplot('Words','freq',data = Freq_word_DF)

ax.set_title("Negative Tweets Words Frequency Plot")



# Neutral Tweets



All_words = []

for sent in TweetData[TweetData['sentiment']=='neutral']['Clean_tweet']:

    for word in word_tokenize(sent):

            All_words.append(word)

All_words_freq = nltk.FreqDist(All_words)

Freq_word_DF = pd.DataFrame({"Data":All_words_freq.most_common(15)})

Freq_word_DF['Words'] = Freq_word_DF['Data'].apply(lambda x : x[0])

Freq_word_DF['freq'] = Freq_word_DF['Data'].apply(lambda x : x[1])

sns.set()

sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})

sns.set_context("paper", font_scale=2)

fig=plt.figure(figsize =(20,8),dpi=50)

ax = sns.barplot('Words','freq',data = Freq_word_DF)

ax.set_title("Neutral Tweets Words Frequency Plot")
# Lets do POS tagging of our clean tweets

def tagging(text):

    tag = nltk.pos_tag(word_tokenize(text))

    return tag
TweetData['Pos_Clean'] = TweetData['Clean_tweet'].apply(lambda x : tagging(x))
TweetData[['Clean_tweet','Pos_Clean']].head()
# Lets See Frequency Distribution of just Adjectives in clean tweet from positive negative and neutal tweets

allowed_word_type = ["JJ","JJR","JJS"]

all_words = []

for pos in TweetData['Pos_Clean']:

    for w in pos:

        if w[1] in allowed_word_type:

            all_words.append(w[0])

All_words_freq = nltk.FreqDist(all_words)

Freq_word_DF = pd.DataFrame({"Data":All_words_freq.most_common(15)})

Freq_word_DF['Words'] = Freq_word_DF['Data'].apply(lambda x : x[0])

Freq_word_DF['freq'] = Freq_word_DF['Data'].apply(lambda x : x[1])

sns.set()

sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})

sns.set_context("paper", font_scale=2)

fig=plt.figure(figsize =(20,8),dpi=50)

ax = sns.barplot('Words','freq',data = Freq_word_DF)

ax.set_title("Adjective words frequncy distribution")
# The code worked above lets check for all types of setiment words:

all_words = []

allowed_word_type = ["JJ","JJR","JJS"]

for pos in TweetData[TweetData['sentiment']=='positive']['Pos_Clean']:

    for w in pos:

        if w[1] in allowed_word_type:

            all_words.append(w[0])

All_words_freq = nltk.FreqDist(all_words)

Freq_word_DF = pd.DataFrame({"Data":All_words_freq.most_common(15)})

Freq_word_DF['Words'] = Freq_word_DF['Data'].apply(lambda x : x[0])

Freq_word_DF['freq'] = Freq_word_DF['Data'].apply(lambda x : x[1])

sns.set()

sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})

sns.set_context("paper", font_scale=2)

fig=plt.figure(figsize =(20,8),dpi=50)

ax = sns.barplot('Words','freq',data = Freq_word_DF)

ax.set_title("Adjective words frequncy distribution for Positive Tweets")



# Negative words

for pos in TweetData[TweetData['sentiment']=='negative']['Pos_Clean']:

    for w in pos:

        if w[1] in allowed_word_type:

            all_words.append(w[0])

All_words_freq = nltk.FreqDist(all_words)

Freq_word_DF = pd.DataFrame({"Data":All_words_freq.most_common(15)})

Freq_word_DF['Words'] = Freq_word_DF['Data'].apply(lambda x : x[0])

Freq_word_DF['freq'] = Freq_word_DF['Data'].apply(lambda x : x[1])

sns.set()

sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})

sns.set_context("paper", font_scale=2)

fig=plt.figure(figsize =(20,8),dpi=50)

ax = sns.barplot('Words','freq',data = Freq_word_DF)

ax.set_title("Adjective words frequncy distribution for Negative Tweets")



# neutral words

for pos in TweetData[TweetData['sentiment']=='neutral']['Pos_Clean']:

    for w in pos:

        if w[1] in allowed_word_type:

            all_words.append(w[0])

All_words_freq = nltk.FreqDist(all_words)

Freq_word_DF = pd.DataFrame({"Data":All_words_freq.most_common(15)})

Freq_word_DF['Words'] = Freq_word_DF['Data'].apply(lambda x : x[0])

Freq_word_DF['freq'] = Freq_word_DF['Data'].apply(lambda x : x[1])

sns.set()

sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})

sns.set_context("paper", font_scale=2)

fig=plt.figure(figsize =(20,8),dpi=50)

ax = sns.barplot('Words','freq',data = Freq_word_DF)

ax.set_title("Adjective words frequncy distribution for Neutral Tweets")
TweetData[TweetData['sentiment']=='negative'].head()
# Lets make some word clouds:

from wordcloud import WordCloud,STOPWORDS 

stopwords = set(STOPWORDS) 

comment_words = ' '

for text in TweetData['Clean_tweet']:

    for words in word_tokenize(text): 

        comment_words = comment_words + words + ' '

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(comment_words)

sns.set()

plt.figure(figsize = (8, 8), facecolor = None,dpi=100) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0)
# Lets make some word clouds for positive Negative and Neural text:

from wordcloud import WordCloud,STOPWORDS 

stopwords = set(STOPWORDS) 

comment_words = ' '

for text in TweetData[TweetData['sentiment']=='positive']['Clean_tweet']:

    for words in word_tokenize(text): 

        comment_words = comment_words + words + ' '

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(comment_words)

sns.set()

plt.figure(figsize = (8, 8), facecolor = None,dpi=100)

plt.title("Positive Tweets word Cloud")

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0)





comment_words = ' '

for text in TweetData[TweetData['sentiment']=='negative']['Clean_tweet']:

    for words in word_tokenize(text): 

        comment_words = comment_words + words + ' '

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(comment_words)

sns.set()

plt.figure(figsize = (8, 8), facecolor = None,dpi=100)

plt.title("Negative Tweets word Cloud")

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0)





comment_words = ' '

for text in TweetData[TweetData['sentiment']=='neutral']['Clean_tweet']:

    for words in word_tokenize(text): 

        comment_words = comment_words + words + ' '

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(comment_words)

sns.set()

plt.figure(figsize = (8, 8), facecolor = None,dpi=100)

plt.title("Neutral Tweets word Cloud")

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0)



# **Looking into the Selected Tweets**
# Cleaning the Selected Tweets:

# stop_words = set(stopwords.words('english'))

def clean_tweets_selected(x):

    # removing the hyperlinks

    clean1 = re.sub('https?://[A-Za-z0-9./]+','',x)

    # removing the hashtags

    clean2 = re.sub('#[A-Za-z0-9]+','',clean1)

    # removing @

    clean3 = re.sub('@[A-Za-z0-9]','',clean2)

    # removing punctuations and lower case conversion

    clean4 = re.sub(r'[^\w\s]','',clean3).lower()

    words = word_tokenize(clean4)

    # removing stopwords

    words = [w for w in words if not w in stop_words]

    sent = ' '.join(words)

    return sent
TweetData['Clean_Selected_Tweets'] = TweetData['selected_text'].apply(lambda x: clean_tweets_selected(x))
TweetData.head()
# All Selected word Freqency curve with out any pos tagging filter

All_words = []

for sent in TweetData['Clean_Selected_Tweets']:

    for word in word_tokenize(sent):

        All_words.append(word)

All_words_freq = nltk.FreqDist(All_words)

Freq_word_DF = pd.DataFrame({"Data":All_words_freq.most_common(15)})

Freq_word_DF['Words'] = Freq_word_DF['Data'].apply(lambda x : x[0])

Freq_word_DF['freq'] = Freq_word_DF['Data'].apply(lambda x : x[1])

sns.set()

sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})

sns.set_context("paper", font_scale=2)

fig=plt.figure(figsize =(20,8),dpi=50)

sns.barplot('Words','freq',data = Freq_word_DF)
# Lets See only selected words with one and two words only

All_words = []

for sent in TweetData['selected_text']:

    if len(word_tokenize(sent))<3:

        for word in word_tokenize(sent):

            All_words.append(word)

All_words_freq = nltk.FreqDist(All_words)

Freq_word_DF = pd.DataFrame({"Data":All_words_freq.most_common(15)})

Freq_word_DF['Words'] = Freq_word_DF['Data'].apply(lambda x : x[0])

Freq_word_DF['freq'] = Freq_word_DF['Data'].apply(lambda x : x[1])

sns.set()

sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})

sns.set_context("paper", font_scale=2)

fig=plt.figure(figsize =(20,8),dpi=50)

sns.barplot('Words','freq',data = Freq_word_DF)
TweetData['Pos_selected_clean'] = TweetData['Clean_Selected_Tweets'].apply(lambda x : tagging(x))
TweetData.head()
Pos = []

for pos in TweetData['Pos_selected_clean']:

    for po in pos:

        Pos.append(po[1])



plt.figure(figsize=(12,6),dpi=100)

sns.set_context(font_scale=.5)

sns.countplot(Pos, facecolor=(0, 0, 0, 0),

                   linewidth=2,

                   edgecolor=sns.color_palette("muted", 2))        



plt.xticks(rotation=70)        
print("number of text with less than 3 selected text length: ",len(TweetData[TweetData['Select_text_words'] < 3]))

Pos = []

for pos in TweetData[TweetData['Select_text_words'] < 3]['Pos_selected_clean']:

    for po in pos:

        Pos.append(po[1])



plt.figure(figsize=(12,6),dpi=100)

sns.set_context(font_scale=.5)

sns.countplot(Pos, facecolor=(0, 0, 0, 0),

                   linewidth=2,

                   edgecolor=sns.color_palette("muted", 2))        



plt.xticks(rotation=70)      
TweetData.head()
selected_pos =  ["JJ","NN","NNS"]

def pos_filter(text):

    con = 0

    for w in text:

        if w[1] in selected_pos:

            con+= 1

    return con
TweetData['count_text'] = TweetData['Pos_Clean'].apply(lambda x: pos_filter(x))

TweetData['Count_selected_text'] = TweetData['Pos_selected_clean'].apply(lambda x: pos_filter(x))
TweetData.head()
# Line plot for both counts above for different tweet sentiments , also for those which has only less that three words in selected text



len(TweetData)
plt.figure(figsize=(12,6),dpi=100)

sns.set_context('paper',font_scale=1)

sns.lineplot(x=TweetData.index[0:50],y = TweetData['count_text'][0:50])

sns.lineplot(x=TweetData.index[0:50],y = TweetData['Count_selected_text'][0:50])
TweetDatasmall =  TweetData[TweetData['Select_text_words'] < 3]

plt.figure(figsize=(12,6),dpi=100)

sns.set_context('paper',font_scale=1)

sns.lineplot(x=TweetDatasmall.index[0:50],y = TweetDatasmall['count_text'][0:50])

sns.lineplot(x=TweetDatasmall.index[0:50],y = TweetDatasmall['Count_selected_text'][0:50])
# live curve for less that 3 words seggregated for difference sentiments :

Positive =  TweetDatasmall[TweetDatasmall['sentiment']=='positive']

Negative =  TweetDatasmall[TweetDatasmall['sentiment']=='negative']

Neutral =   TweetDatasmall[TweetDatasmall['sentiment']=='neutral']



plt.figure(figsize=(12,12),dpi=100)

plt.subplot(3,1,1)

sns.set_context('paper',font_scale=1)

sns.lineplot(x=Positive.index[0:50],y = Positive['count_text'][0:50])

sns.lineplot(x=Positive.index[0:50],y = Positive['Count_selected_text'][0:50])

plt.title("Positive tweets count Line Plots")

plt.subplot(3,1,2)

sns.set_context('paper',font_scale=1)

sns.lineplot(x=Negative.index[0:50],y = Negative['count_text'][0:50])

sns.lineplot(x=Negative.index[0:50],y = Negative['Count_selected_text'][0:50])

plt.title("Negative tweets count Line Plots")

plt.subplot(3,1,3)

sns.set_context('paper',font_scale=1)

sns.lineplot(x=Neutral.index[0:50],y = Neutral['count_text'][0:50])

sns.lineplot(x=Neutral.index[0:50],y = Neutral['Count_selected_text'][0:50])

plt.title("Neutral tweets count Line Plots")
# For N grams first I would like to Clean the tweets with out removing stop words

def clean_tweets_N(x):

    # removing the hyperlinks

    clean1 = re.sub('https?://[A-Za-z0-9./]+','',x)

    # removing the hashtags

    clean2 = re.sub('#[A-Za-z0-9]+','',clean1)

    # removing @

    clean3 = re.sub('@[A-Za-z0-9]','',clean2)

    # removing punctuations and lower case conversion

    clean4 = re.sub(r'[^\w\s]','',clean3).lower()

#     words = word_tokenize(clean4)

#     # removing stopwords

#     words = [w for w in words if not w in stop_words]

#     sent = ' '.join(words)

    return clean4
TweetData['Clean_for_N-gram'] = TweetData['text'].apply(lambda x : clean_tweets_N(x))
from nltk.util import ngrams

def extract_ngrams(data, num):

    n_grams = ngrams(nltk.word_tokenize(data), num)

    return [ ' '.join(grams) for grams in n_grams]
TweetData['Ngram-2'] = TweetData['Clean_for_N-gram'].apply(lambda x : extract_ngrams(x,2))

TweetData['Ngram-3'] = TweetData['Clean_for_N-gram'].apply(lambda x : extract_ngrams(x,3))
TweetData.head()
# Plotting Ngram-2

All_ngram = []

for sent in TweetData['Ngram-2']:

    for gram in sent:

        All_ngram.append(gram)

All_ngram_freq = nltk.FreqDist(All_ngram)

Freq_word_DF = pd.DataFrame({"Data":All_ngram_freq.most_common(15)})

Freq_word_DF['2-Grams'] = Freq_word_DF['Data'].apply(lambda x : x[0])

Freq_word_DF['freq'] = Freq_word_DF['Data'].apply(lambda x : x[1])

sns.set()

# sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})

sns.set_context("paper", font_scale=1)

fig=plt.figure(figsize =(12,8),dpi=100)

sns.barplot('2-Grams','freq',data = Freq_word_DF,dodge=False)
# Plotting Ngram-3

All_ngram = []

for sent in TweetData['Ngram-3']:

    for gram in sent:

        All_ngram.append(gram)

All_ngram_freq = nltk.FreqDist(All_ngram)

Freq_word_DF = pd.DataFrame({"Data":All_ngram_freq.most_common(15)})

Freq_word_DF['3-Grams'] = Freq_word_DF['Data'].apply(lambda x : x[0])

Freq_word_DF['freq'] = Freq_word_DF['Data'].apply(lambda x : x[1])

sns.set()

# sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})

sns.set_context("paper", font_scale=1)

fig=plt.figure(figsize =(12,8),dpi=100)

sns.barplot('3-Grams','freq',data = Freq_word_DF,dodge=False)

plt.xticks(rotation=70)  