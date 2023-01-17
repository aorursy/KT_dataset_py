def bigramEstimation(file):

    '''A very basic solution for the sake of illustration.

       It can be calculated in a more sophesticated way.

       '''



    lst = [] # This will contain the tokens

    unigrams = {} # for unigrams and their counts

    bigrams = {} # for bigrams and their counts



    # 1. Read the textfile, split it into a list

    #text = open(file, 'r').read()

    #text= open(file, encoding ='utf-8')

    #lst = text.strip().split()

    

    

    with open(file) as f:

        lst = list(f)

    

    

    

    print('Read ', len(lst), ' tokens...')



    #del text # No further need for text var







    # 2. Generate unigrams frequencies

    for l in lst:

        if not l in unigrams:

            unigrams[l] = 1

        else:

            unigrams[l] += 1



    print('Generated ', len(unigrams), ' unigrams...') 



    # 3. Generate bigrams with frequencies

    for i in range(len(lst) - 1):

        temp = (lst[i], lst[i+1]) # Tuples are easier to reuse than nested lists

        if not temp in bigrams:

            bigrams[temp] = 1

        else:

            bigrams[temp] += 1



    print('Generated ', len(bigrams), ' bigrams...')



    # Now Hidden Markov Model

    # bigramProb = (Count(bigram) / Count(first_word)) + (Count(first_word)/ total_words_in_corpus)

    # A few things we need to keep in mind

    total_corpus = sum(unigrams.values())

    # You can add smoothed estimation if you want





    print('Calculating bigram probabilities and saving to file...')



    # Comment the following 4 lines if you do not want the header in the file. 

    with open("bigrams.txt", 'a') as out:

        out.write('Bigram' + '\t' + 'Bigram Count' + '\t' + 'Uni Count' + '\t' + 'Bigram Prob')

        out.write('\n')

        out.close()





    for k,v in bigrams.items():

        # first_word = helle in ('hello', 'world')

        first_word = k[0]

        first_word_count = unigrams[first_word]

        bi_prob = bigrams[k] / unigrams[first_word]

        uni_prob = unigrams[first_word] / total_corpus



        final_prob = bi_prob + uni_prob

        with open("bigrams.txt", 'a') as out:

            out.write(k[0] + ' ' + k[1] + '\t' + str(v) + '\t' + str(first_word_count) + '\t' + str(final_prob)) # Delete whatever you don't want to print into a file

            out.write('\n')

            out.close()
inputSentence= input("Please Enter your text:")
txt=inputSentence
#remove punctuations and standardize text

words=txt.split()

words
#lowercasing all words

words = [word.lower() for word in words]

words
import string

table = str.maketrans('', '', string.punctuation)

stripped = [w.translate(table) for w in words]

print(stripped)
#cleaning data using nltk library

!pip3 install nltk

!python3 -m nltk.downloader all
from nltk.tokenize import word_tokenize

tokens = word_tokenize(txt)

# convert to lower case

tokens = [w.lower() for w in tokens]

# remove punctuation from each word

import string

table = str.maketrans('', '', string.punctuation)

stripped = [w.translate(table) for w in tokens]

# remove remaining tokens that are not alphabetic

words = [word for word in stripped if word.isalpha()]

# filter out stop words

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

words = [w for w in words if not w in stop_words]
#create unigram aka frequencies

import numpy as np

import matplotlib.pyplot as plt



data= np.array(stripped)
from itertools import groupby

freq = {key:len(list(group)) for key, group in groupby(np.sort(data))}
freq
type(freq)



dict=freq
unigrams=[]

for i,j in freq.items():

    unigrams.append ((i,j))

print (unigrams)
words
i_len=len(stripped)-1



bi_list=[]

for i in range(0,i_len):

    bi_list.append(stripped[i]+" "+stripped[i+1])

bi_list

freq = {key:len(list(group)) for key, group in groupby(np.sort(bi_list))}



bigrams=[]

for i,j in freq.items():

    bigrams.append ((i,j))

print (bigrams)

i_len=len(stripped)-1



for i in range(0,i_len):

    print(stripped[i]+" "+stripped[i+1])

    print("P("+stripped[i+1]+'|'+stripped[i]+')=')

print(bigrams[5])
bigrams