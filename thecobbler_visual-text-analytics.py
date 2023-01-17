#This is a usual set of Visualizations we can use while working with Text data.

#The idea was to build a repository for future correspondence

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from wordcloud import WordCloud,STOPWORDS

import nltk
# import the dataset



from nltk.corpus import inaugural

# extract the datataset in raw format, you can also extract it in other formats as well

text = inaugural.raw()

wordcloud = WordCloud(max_font_size=60).generate(text)

plt.figure(figsize=(16,12))

# plot wordcloud in matplotlib

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
from  nltk.book import text4 as inaugural_speeches

plt.figure(figsize=(16,5))

topics = ['citizens', 'democracy', 'freedom', 'duties', 'America','principle','people', 'Government']

inaugural_speeches.dispersion_plot(topics)
from nltk.corpus import brown

stop_words = set(STOPWORDS)

topics = ['government', 'news', 'religion','adventure','hobbies']

for topic in topics:

    # filter out stopwords and punctuation mark and only create array of words

    words = [word for word in brown.words(categories=topic)

            if word.lower() not in stop_words and word.isalpha() ]

    freqdist = nltk.FreqDist(words)

    # print 5 most frequent words

    print(topic,'more :', ' , '.join([ word.lower() for word, count in freqdist.most_common(5)]))

    # print 5 least frequent words

    print(topic,'less :', ' , '.join([ word.lower() for word, count in freqdist.most_common()[-5:]]))

# get all words for government corpus

corpus_genre = 'government'

words = [word for word in brown.words(categories=corpus_genre) if word.lower() not in stop_words and word.isalpha() ]

freqdist = nltk.FreqDist(words)

plt.figure(figsize=(16,5))

freqdist.plot(50)
def lexical_diversity(text):

    return round(len(set(text)) / len(text),2) #Measure of uniqueness



def get_brown_corpus_words(category, include_stop_words=False):

    '''helper method to get word array for a particular category

     of brown corpus which may/may not include the stopwords that can be toggled

     with the include_stop_words flag in the function parameter'''

    if include_stop_words:

        words = [word.lower() for word in brown.words(categories=category) if word.isalpha() ]

    else:

        words = [word.lower() for word in brown.words(categories=category)

                 if word.lower() not in stop_words and word.isalpha() ]

    return words



# calculate and print lexical diversity for each genre of the brown corpus

for genre in brown.categories():

    lex_div_with_stop = lexical_diversity(get_brown_corpus_words(genre, True))

    lex_div = lexical_diversity(get_brown_corpus_words(genre, False))

    print(genre ,lex_div , lex_div_with_stop)
#Function to sort the words of a given corpus and category lexicographically

def Lexo_sort(corpus,category):

    words1 = sorted([wrd for wrd in list(set(corpus.words(categories=category))) if wrd.isalpha()])

    return (words1) 
cfd = nltk.ConditionalFreqDist(

           (genre, len(word))

           for genre in brown.categories()

           for word in get_brown_corpus_words(genre))



plt.figure(figsize=(16,8))

cfd.plot()
from nltk.util import ngrams

plt.figure(figsize=(16,8))

for genre in brown.categories():

    sol = []

    for i in range(1,6):

        count = 0

        fdist = nltk.FreqDist(ngrams(get_brown_corpus_words(genre), i))

        sol.append(len([cnt for ng,cnt in fdist.most_common() if cnt > 1]))

    plt.plot(np.arange(1,6), sol, label=genre)

plt.legend()

plt.show()