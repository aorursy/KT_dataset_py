import numpy as np

import pandas as pd

import nltk

import re

import string

from nltk.tokenize import word_tokenize

from subprocess import check_output

from nltk.stem import WordNetLemmatizer

from bs4 import BeautifulSoup

import matplotlib.pyplot as plt
physic = pd.read_csv("../input/test.csv")
physic.head(5)
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

    

    #print('Lemmatizing')

    wordnet_lemmatizer = WordNetLemmatizer()

    data = data.apply(lambda x: [wordnet_lemmatizer.lemmatize(i) for i in x])

    data = data.apply(lambda x: [i for i in x if len(i)>2])

    return(data)
#nltk.download()
def get_frequency(title):

    

    frequency = []

    inverse_frequency = {}

    for i in range(len(title)):

        word_count = {}



        for word in title[i]:

            if word in word_count:    

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
title = data_clean(physic.title)
frequency, inverse_frequency = get_frequency(title)
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

    
index = 1
sorted(tfidf[index].items(), key=operator.itemgetter(1), reverse=True)
print(physic.title[index])

print(physic.content[index])
tfidf_distribution = sorted(tfidf_distribution)

print(len(tfidf_distribution))
plt.plot(tfidf_distribution)

plt.show()
plt.plot(np.log(tfidf_distribution))

plt.show()
top = 8

output = []

for i in range(0,len(physic)):

    prediction = sorted(tfidf[i], key=tfidf[i].get, reverse=True)[0:top]

    output.append([physic.id[i], ' '.join(prediction)])
pd.DataFrame(data=output,columns = ['id','tags']).to_csv('Submission.csv', index=False)       