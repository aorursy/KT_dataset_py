## importing our libraries

import pandas as pd

import numpy as np



## we are goingo to use bigram and trigrams to separate our dataset in combinations of two and three words

from nltk.util import bigrams,trigrams



## we are going to use Counter to counter the bigrams and trigrams of our dataset.

from collections import Counter



## let's use regular expressions to treat our dataset

import re



## let's use random to choose the words of our poem based on a probability distribution

import random
## reading the dataset

df = pd.read_csv('/kaggle/input/poems-in-portuguese/portuguese-poems.csv')



## droping any NA values from content (that's the columns where the content of the poem is written)

df.dropna(subset=['Content'],inplace=True)



## reseting the index

df.reset_index(drop=True,inplace=True)

df.shape
## checking the distribution of authors



df.Author.value_counts(normalize=True)[:10].plot.bar();
## let's generate our corpus



def generate_corpus(df):

    

    corpus = []

    

    for i in range(len(df)):

        s = df.Content[i].casefold().replace('\r\n', ' ').replace('\n',' ')

        s = re.sub('([.,!?():-])', r' \1 ', s)

        s = re.sub('\s{2,}', ' ', s).split(' ')

        corpus.append(s)

        

    return corpus



## creating the tags for our corpus



def create_tags(corpus):

    

    words = []

    

    for sentence in corpus:

        sentence.insert(0, '<s>')

        sentence.insert(0, '<s>')

        sentence.append('</s>')

        words.extend(sentence)

        

    return words
## creating the corpus



corpus = generate_corpus(df)

print(corpus[0:2])
## inserting the tags and gerenating a single string to divide our dataset in bigrams and trigrams



single_string = create_tags(corpus)

print(single_string[0:100])
## creating the bigrams of each combination of word



words_bigram = bigrams(single_string)
## creating the trigram of each combination of word



words_trigram = trigrams(single_string)
## counting bigrams



bigram_count = Counter(words_bigram)

bigram_count[('<s>', '<s>')]
## counting trigrams



trigram_count = Counter(words_trigram)

trigram_count[('<s>', '<s>','eu')]
## creating a list of bigramns and trigrams so we can calculate the probilities



bigram_key = list(bigram_count.keys())

trigram_key = list(trigram_count.keys())
list_bigram = []

list_lastword = []

list_probs = []



## for each trigram t



for t in trigram_key:

    

    ## create a bigram using the first two words of the trigram

    key_bigram = (t[0], t[1])

    

    ## find how many times the trigram happened and divide it by the number of times that the bigram happened

    prob = trigram_count[t] / bigram_count[key_bigram]

    

    ## append the lists above

    list_bigram.append(key_bigram)

    list_lastword.append(t[2])

    list_probs.append(prob)
## creating a dataframe with the results



model_df = pd.DataFrame({'bigram': list_bigram, 'lastword': list_lastword, 'prob': list_probs})

model_df
## how many time <s><s> appeared



bigram_count[('<s>', '<s>')]
## how many time <s><s><eu> appeared



trigram_count[('<s>', '<s>','eu')]
## probability of the word 'eu' appearing next to <s><s>



319 / 15541
## taking the proof



model_df.iloc[0,:]
test_df = model_df.loc[model_df['bigram'] == ('<s>', '<s>')]

test_df.sort_values('prob',ascending=False)
test_df = model_df.loc[model_df['bigram'] == ('<s>', 'a')]

test_df.sort_values('prob',ascending=False)
num_sents = 2

current_bigram = ('<s>', '<s>')

i = 0

while i < num_sents:

    df = model_df.loc[model_df['bigram'] == current_bigram]

    words = df['lastword'].values

    probs = df['prob'].values

    last_word = random.choices(words, probs)[0]

    

    current_bigram = (current_bigram[1], last_word)

    

    if last_word == '</s>':

        i+=1

    

    if last_word != '<s>' and last_word != '</s>':

        print(last_word, end=' ')