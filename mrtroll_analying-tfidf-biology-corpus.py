import numpy as np

import pandas as pd

import nltk

import re

import string

from nltk.corpus import stopwords

from nltk.util import ngrams

from nltk.tokenize import word_tokenize

from subprocess import check_output

from nltk.stem import WordNetLemmatizer

from bs4 import BeautifulSoup

import matplotlib.pyplot as plt

print(check_output(["ls", "../input"]).decode("utf8"))
biology = pd.read_csv("../input/biology.csv")
biology.head(5)
swords1 = stopwords.words('english')



punctuations = string.punctuation



def data_clean(data):

    print('Cleaning data')

    data = data.apply(lambda x: x.lower())

    data = data.apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())

    data = data.apply(lambda x: re.sub(r'^\W+|\W+$',' ',x))

    data = data.apply(lambda i: ''.join(i.strip(punctuations))  )

    #print('tokenize')

    data = data.apply(lambda x: word_tokenize(x))



    #Select only the nouns

    is_noun = lambda pos: pos[:2] == 'NN' 

    for i in range(len(data)):

        data[i] = [word for (word, pos) in nltk.pos_tag(data[i]) if is_noun(pos)]

    

    #print('Remove stopwords')

    data = data.apply(lambda x: [i for i in x if i not in swords1 if len(i)>2])

    #print('minor clean some wors')

    data = data.apply(lambda x: [i.split('/') for i in x] )

    data = data.apply(lambda x: [i for y in x for i in y])

    #print('Lemmatizing')

    wordnet_lemmatizer = WordNetLemmatizer()

    data = data.apply(lambda x: [wordnet_lemmatizer.lemmatize(i) for i in x])

    data = data.apply(lambda x: [i for i in x if len(i)>2])

    return(data)
#nltk.download()
def get_frequency(content, title):

    

    frequency = []

    inverse_frequency = {}

    for i in range(len(content)):

        word_count = {}

        important = {}

        for word in title[i]:

            if word in word_count:

                word_count[word] = word_count[word] + 100

            else:

                word_count[word] = 10

                important[word] = True

        for word in content[i]:

            if word in word_count:

                if word in important:

                    word_count[word] = word_count[word] + 50

                else:

                    word_count[word] = word_count[word] + 1

            else:

                word_count[word] = 1

                

        for word in word_count:

            if word in inverse_frequency:

                inverse_frequency[word] = inverse_frequency[word] + 1

            else:

                inverse_frequency[word] = 1            

        frequency.append(word_count)

    return (frequency, inverse_frequency)
content = data_clean(biology.content)

title = data_clean(biology.title)
frequency, inverse_frequency = get_frequency(content, title)
import operator

frequency_words = {}

for document in frequency:

    for word in document:

        if word in frequency_words:

            frequency_words[word] = frequency_words[word] + document[word]

        else:

            frequency_words[word] = document[word]            

frequency_words = sorted(frequency_words.values())
print('number of words:',len(frequency_words))
plt.plot(frequency_words)

plt.show()
plt.plot(np.log(frequency_words))

plt.show()
tfidf = frequency
tfidf_distribution = []

for document in tfidf:

    if document == {}:

        continue

    max_frequency = sorted(document.items(), key=operator.itemgetter(1), reverse=True)[0][1]

    for word in document:

        document[word] = document[word]/(max_frequency + 0.0)*np.log(len(tfidf)/(inverse_frequency[word]+0.))

        tfidf_distribution.append(document[word])

    
index = 0
sorted(tfidf[index].items(), key=operator.itemgetter(1), reverse=True)
print(biology.title[index])

print(biology.content[index])

print(biology.tags[index])
tfidf_distribution = sorted(tfidf_distribution)

print(len(tfidf_distribution))
plt.plot(tfidf_distribution)

plt.show()
plt.plot(np.log(tfidf_distribution))

plt.show()
def getF1(prediction,tags):

    if len(prediction) == 0 or len(tags) == 0:

        return 0.0

    tags = set(tags.split())

    corrects = 0

    for p in prediction:

        if p in tags:

            corrects = corrects + 1

    

    precision = corrects / (len(prediction) + 0.)

    recall = corrects / (len(tags) + 0.)

    if precision == 0 or recall == 0:

        return 0.0     

    return 2*precision*recall/(precision + recall)

        
top = 3

corpusf1 = []

for i in range(len(tfidf)):

    prediction = sorted(tfidf[i], key=tfidf[i].get, reverse=True)[0:top]

    corpusf1.append(getF1(prediction, biology.tags[i]))
print(np.average(corpusf1))