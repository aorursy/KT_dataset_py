#Importing libraries

import nltk, re, pprint

import numpy as np

import pandas as pd

import requests

import matplotlib.pyplot as plt

import seaborn as sns

import pprint, time

import random

from sklearn.model_selection import train_test_split

from nltk.tokenize import word_tokenize
# reading the Treebank tagged sentences

wsj = list(nltk.corpus.treebank.tagged_sents())
# first few tagged sentences

wsj[:3]
# Splitting into train and test

random.seed(1234)

train_set, test_set = train_test_split(wsj,test_size=0.2)



print(len(train_set))

print(len(test_set))

print(train_set[:40])
# Getting list of tagged words from train set

train_tagged_words = [(word,tag) for sent in train_set for word,tag in sent]

len(train_tagged_words)
# tokens in train set

tokens = [word for word,tag in train_tagged_words]

tokens[:10]
# unique vocabulary from train set

V = set(tokens)

print(len(V))
# number of tags in train set

T = set([tag for word,tag in train_tagged_words])

len(T)
# . -> represent the start of a sentence

# $ -> represent the end of the sentence

print(T)
t = len(T)

v = len(V)

w_given_t = np.zeros((t, v))

w_given_t.shape
# compute word given tag: Emission Probability

# p(w|t) = (#word w tagged with tag t in the corpus) / (#tag t appearing in the corpus)



def word_given_tag(word, tag, train_bag = train_tagged_words):

    tag_list = [(w,t) for w,t in train_bag if t==tag]

    count_tag = len(tag_list) # count of tag t present in the corpus

    w_given_tag_list = [w for w,t in tag_list if w==word] #word w with the tag t present in the corpus

    count_w_given_tag = len(w_given_tag_list) #count of word w with the tag t in the corpus

    

    return (count_w_given_tag, count_tag)
# examples



# large

print("\n", "large")

print(word_given_tag('large', 'JJ'))

print(word_given_tag('large', 'VB'))

print(word_given_tag('large', 'NN'), "\n")



# will

print("\n", "will")

print(word_given_tag('will', 'MD'))

print(word_given_tag('will', 'NN'))

print(word_given_tag('will', 'VB'))



# book

print("\n", "book")

print(word_given_tag('book', 'NN'))

print(word_given_tag('book', 'VB'))



# Android

print("\n", "android")

print(word_given_tag('android', 'NN'))

"""word_with_tag_matrix = np.zeros((len(T), len(V)), dtype='float32')

for i, t in enumerate(list(T)):

    for j, w in enumerate(list(V)): 

        word_with_tag_matrix[i, j] = word_given_tag(w, t)[0]/word_given_tag(w, t)[1]"""
# convert the matrix to a df for better readability

#word_with_tags_df = pd.DataFrame(word_with_tag_matrix, columns = list(V), index=list(T))

#word_with_tags_df
# compute tag given tag: tag2(t2) given tag1 (t1), i.e. Transition Probability

# p(t2|t1)= (#tag t1 is followed by tag t2)/ (#tag t1 appearing in corpus)



def t2_given_t1(t2, t1, train_bag = train_tagged_words):

    tags = [t for w,t in train_bag] #get all the tags from training set

    count_t1 = len([t for t in tags if t==t1]) #count of t1 appearing in the corpus

    count_t2_t1 = 0  #count of t2 coming after t1 -> t1 followed by t2

    for index in range(len(tags)-1):

        if tags[index]==t1 and tags[index+1] == t2:

            count_t2_t1 += 1 #increment count if t1 is followed by t2

    return (count_t2_t1, count_t1)
def t2_given_t1_prob(t2,t1,train_bag = train_tagged_words):

    count_t2_t1, count_t1 = t2_given_t1(t2,t1,train_bag)

    return count_t2_t1/count_t1
# examples

print(t2_given_t1(t2='NNP', t1='JJ'))

print(t2_given_t1('NN', 'JJ'))

print(t2_given_t1('NN', 'DT'))

print(t2_given_t1('NNP', 'VB'))

print(t2_given_t1(',', 'NNP'))

print(t2_given_t1('PRP', 'PRP'))

print(t2_given_t1('VBG', 'NNP'))

print(t2_given_t1('VB', 'MD'))
#Please note P(tag|start) is same as P(tag|'.')

print(t2_given_t1('DT', '.'))

print(t2_given_t1('VBG', '.'))

print(t2_given_t1('NN', '.'))

print(t2_given_t1('NNP', '.'))
# creating t x t transition matrix of tags

# each column is t2, each row is t1

# thus M(i, j) represents P(tj given ti)



tags_matrix = np.zeros((len(T), len(T)), dtype='float32')

for i, t1 in enumerate(list(T)): 

    for j, t2 in enumerate(list(T)): 

        tags_matrix[i, j] = t2_given_t1(t2, t1)[0]/t2_given_t1(t2, t1)[1]
# convert the matrix to a df for better readability

tags_df = pd.DataFrame(tags_matrix, columns = list(T), index=list(T))

tags_df #row->t1, col->t2
# Let's see the prob for tags appearing at start of the sentence represented by tag .

tags_df.loc['.', :]
# heatmap of tags matrix

# T(i, j) means P(tag j given tag i)

plt.figure(figsize=(18, 12))

sns.heatmap(tags_df)

plt.show()

# frequent tags

# filter the df to get P(t2, t1) > 0.5

tags_frequent = tags_df[tags_df>0.5]

plt.figure(figsize=(18, 12))

sns.heatmap(tags_frequent)

plt.show()
len(train_tagged_words)
# Viterbi Heuristic

def Viterbi(words, train_bag = train_tagged_words):

    state = [] #state/tag for each word

    T = list(set([tag for word,tag in train_bag])) #tags in the corpus

    

    for index, word in enumerate(words):

        #initialise list of probability column for a given observation

        state_probalities = [] #prob for each state/word in corpus for each word

        for t2 in T:

            if index == 0:

                transition_p = t2_given_t1_prob(t2, '.') #transition prob. for start tag

            else:

                t1 = state[-1]

                transition_p = t2_given_t1_prob(t2,t1) #transition prob. for tag t1 followed by t2

                

            # compute emission and state probabilities

            emission_p = word_given_tag(words[index], t2)[0]/word_given_tag(words[index], t2)[1]  # p(w|tag) -> count of word with tag t2 / total number of t2

            state_probalities.append(emission_p * transition_p) 

            

        # getting state for which probability is maximum

        state_with_max_prob = T[state_probalities.index(max(state_probalities))] 

        state.append(state_with_max_prob)

    return list(zip(words, state))



# Running on entire test dataset would take more than 3-4hrs. 

# Let's test our Viterbi algorithm on a few sample sentences of test dataset



random.seed(100)



# choose random 5 sents index from test set

rndom_index = [random.randint(1,len(test_set)) for x in range(50)]



# get the 5 sent from test set using the 5 random_index we picked above

test_run = [test_set[i] for i in rndom_index]



# list of tagged words - this we will use for evaluation purpose

test_run_base = [(word,tag) for sent in test_run for word,tag in sent]



# list of untagged words

test_words = [word for word,tag in test_run_base] 

test_words
# tagging the test sentences

start = time.time()

tagged_seq = Viterbi(test_words)

end = time.time()

difference = end-start
print("Time taken in seconds: ", difference)

print(tagged_seq)

#print(test_run_base)
# accuracy

check = [(i,j) for i, j in zip(tagged_seq, test_run_base) if i == j] 

accuracy = len(check)/len(tagged_seq)

accuracy
incorrect_tagged_cases = [(test_run_base[tagged_seq.index(i)-1],i,j) for i, j in zip(tagged_seq, test_run_base) if i != j] 

incorrect_tagged_cases
## Testing

sentence_test = 'Twitter is the best networking social site. Man is a social animal. Data science is an emerging field. Data science jobs are high in demand.'

words = word_tokenize(sentence_test)



start = time.time()

tagged_seq = Viterbi(words)

end = time.time()

difference = end-start
print(tagged_seq)

print(difference)
sentence = "Donald Trump is the current President of US. Before entering politics, he was a domineering businessman and television personality."

words = word_tokenize(sentence)



tagged_seq = Viterbi(words)

tagged_seq