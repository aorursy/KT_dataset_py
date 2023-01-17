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

nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))
# first few tagged sentences

print(nltk_data[:10])
# Splitting into train and test

random.seed(1234)

train_set, test_set = train_test_split(nltk_data,test_size=0.05)



print(len(train_set))

print(len(test_set))

print(train_set[:10])
# Getting list of tagged words

train_tagged_words = [tup for sent in train_set for tup in sent]

len(train_tagged_words)
# tokens 

tokens = [pair[0] for pair in train_tagged_words]

tokens[:10]
# vocabulary

V = set(tokens)

print(len(V))
# number of tags

T = set([pair[1] for pair in train_tagged_words])

len(T)
print(T)
len(train_tagged_words)
# computing P(w/t) and storing in T x V matrix

t = len(T)

v = len(V)

w_given_t = np.zeros((t, v))
# compute word given tag: Emission Probability

def word_given_tag(word, tag, train_bag = train_tagged_words):

    tag_list = [pair for pair in train_bag if pair[1]==tag]

    count_tag = len(tag_list)

    w_given_tag_list = [pair[0] for pair in tag_list if pair[0]==word]

    count_w_given_tag = len(w_given_tag_list)

    

    return (count_w_given_tag, count_tag)
# examples



# large

print("\n", "large")

print(word_given_tag('large', 'ADJ'))

print(word_given_tag('large', 'VERB'))

print(word_given_tag('large', 'NUM'), "\n")



# will

print("\n", "will")

print(word_given_tag('will', 'VERB'))

print(word_given_tag('will', 'PRT'))

print(word_given_tag('will', 'DET'))

print(word_given_tag('will', 'X'))

print(word_given_tag('will', 'CONJ'))

print(word_given_tag('will', 'PRON'))

print(word_given_tag('will', 'ADJ'))

print(word_given_tag('will', 'ADV'))

print(word_given_tag('will', 'ADP'))  

print(word_given_tag('will', 'X'))        

print(word_given_tag('will', '.'))  

print(word_given_tag('will', 'NUM'))        

      

# book

print("\n", "book")

print(word_given_tag('book', 'NOUN'))

print(word_given_tag('book', 'VERB'))
### Transition Probabilities
# compute tag given tag: tag2(t2) given tag1 (t1), i.e. Transition Probability



def t2_given_t1(t2, t1, train_bag = train_tagged_words):

    tags = [pair[1] for pair in train_bag]

    count_t1 = len([t for t in tags if t==t1])

    count_t2_t1 = 0

    for index in range(len(tags)-1):

        if tags[index]==t1 and tags[index+1] == t2:

            count_t2_t1 += 1

    return (count_t2_t1, count_t1)
# examples

print(t2_given_t1(t2='NOUN', t1='ADJ'))

print(t2_given_t1('NOUN', 'DET'))

print(t2_given_t1('NOUN', 'NUM'))

print(t2_given_t1('NOUN', 'VERB'))

print(t2_given_t1(',', 'NOUN'))

print(t2_given_t1('PRON', 'PRON'))

print(t2_given_t1('VERB', 'NOUN'))

print(t2_given_t1('ADP', 'VERB'))

print(t2_given_t1('PRT', 'VERB'))

print(t2_given_t1('NOUN', 'CONJ'))

print(t2_given_t1('NOUN', 'ADP'))

print(t2_given_t1('VERB', 'PRON'))

print(t2_given_t1('PRON', 'DET'))
#Please note P(tag|start) is same as P(tag|'.')

print(t2_given_t1('DET', '.'))

print(t2_given_t1('VERB', '.'))

print(t2_given_t1('NOUN', '.'))

print(t2_given_t1('ADJ', '.'))

print(t2_given_t1('PRON', '.'))

print(t2_given_t1('NUM', '.'))

print(t2_given_t1('PRT', '.'))

print(t2_given_t1('X', '.'))

print(t2_given_t1('ADP', '.'))

print(t2_given_t1('ADV', '.'))

print(t2_given_t1('CONJ', '.'))
# creating t x t transition matrix of tags

# each column is t2, each row is t1

# thus M(i, j) represents P(tj given ti)



tags_matrix = np.zeros((len(T), len(T)), dtype='float32')

for i, t1 in enumerate(list(T)):

    for j, t2 in enumerate(list(T)): 

        tags_matrix[i, j] = t2_given_t1(t2, t1)[0]/t2_given_t1(t2, t1)[1]
# convert the matrix to a df for better readability

tags_df = pd.DataFrame(tags_matrix, columns = list(T), index=list(T))
tags_df
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
# Viterbi Heuristic

def Viterbi(words, train_bag = train_tagged_words):

    state = []

    T = list(set([pair[1] for pair in train_bag]))

    for key, word in enumerate(words):

        #initialise list of probability column for a given observation

        p = [] 

        for tag in T:

            if key == 0:

                transition_p = tags_df.loc['.', tag]

            else:

                transition_p = tags_df.loc[state[-1], tag]

                

            # compute emission and state probabilities

            emission_p = word_given_tag(words[key], tag)[0]/(word_given_tag(words[key], tag)[1])

            state_probability = emission_p * transition_p    

            p.append(state_probability)

            

        pmax = max(p)

        # getting state for which probability is maximum

        state_max = T[p.index(pmax)] 

        state.append(state_max)

    return list(zip(words, state))
## Evaluating on Test Set



# Running on entire test dataset would take more than 3-4hrs. 

# Let's test our Viterbi algorithm on a few sample sentences of test dataset



random.seed(1234)



# choose random 5 sents

#rndom = [random.randint(1,len(test_set)) for x in range(5)]



# list of sents

#test_run = [test_set[i] for i in rndom]

test_run=test_set

# list of tagged words

test_run_base = [tup for sent in test_run for tup in sent]



# list of untagged words

test_tagged_words = [tup[0] for sent in test_run for tup in sent]

test_run
len(test_tagged_words)
# tagging the test sentences

start = time.time()

tagged_seq = Viterbi(test_tagged_words)

end = time.time()

difference = end-start

print("Time taken in seconds: ", difference)
#print("Time taken in seconds: ", difference)

print(tagged_seq)
# accuracy

check = [i for i, j in zip(tagged_seq, test_run_base) if i == j] 
accuracy = len(check)/len(tagged_seq)
accuracy
incorrect_tagged_cases = [[test_run_base[i-1],j] for i, j in enumerate(zip(tagged_seq, test_run_base)) if j[0]!=j[1]]
incorrect_tags =[j for i, j in enumerate(zip(tagged_seq, test_run_base)) if j[0]!=j[1]]

incorrect_words = [w[0] for in_wo in incorrect_tags for w in in_wo]

set(incorrect_words)
incorrect_words = [w[0] for in_wo in incorrect_tagged_cases for w in in_wo]

set(incorrect_words)
## Testing

sentence_test = 'Android has been the best-selling OS worldwide on smartphones since 2011 and on tablets since 2013.'

words = word_tokenize(sentence_test)

start = time.time()

tagged_seq = Viterbi(words)

end = time.time()

difference = end-start
print(tagged_seq)

print(difference)
## Testing

sentence_test = 'Android is a mobile operating system developed by Google.'

words = word_tokenize(sentence_test)

start = time.time()

tagged_seq = Viterbi(words)

end = time.time()

difference = end-start
print(tagged_seq)

print(difference)
## Testing

sentence_test = 'Google and Twitter made a deal in 2015 that gave Google access to Twitter s firehose.'

words = word_tokenize(sentence_test)

start = time.time()

tagged_seq = Viterbi(words)

end = time.time()

difference = end-start

print(tagged_seq)

print(difference)


def Viterbi_modified_smoothing(words, train_bag = train_tagged_words):

    state = []

    T = list(set([pair[1] for pair in train_bag]))

    V= len(list(set([pair[0] for pair in words])))

    print(V)  

    for key, word in enumerate(words):

        #initialise list of probability column for a given observation

        p = [] 

        for tag in T:

            if key == 0:

                transition_p = tags_df.loc['.', tag]

            else:

                transition_p = tags_df.loc[state[-1], tag]

                

            # compute emission and state probabilities

           

            emission_p = word_given_tag(words[key], tag)[0]+1/(word_given_tag(words[key], tag)[1]+V)

                         

            state_probability = emission_p * transition_p    

            p.append(state_probability)

            

        pmax = max(p)

        # getting state for which probability is maximum

        state_max = T[p.index(pmax)] 

        state.append(state_max)

    return list(zip(words, state))

## Evaluating on Test Set



# Running on entire test dataset would take more than 3-4hrs. 

# Let's test our Viterbi algorithm on a few sample sentences of test dataset



random.seed(1345)



# choose random 20 sents

#rndom = [random.randint(1,len(test_set)) for x in range(30)]



# list of sents

#test_run1 = [test_set[i] for i in rndom]

test_run1=test_set

# list of tagged words

test_run_base1 = [tup for sent in test_run1 for tup in sent]



# list of untagged words

test_tagged_words1 = [tup[0] for sent in test_run1 for tup in sent]

test_run1
len(test_tagged_words1)
# tagging the test sentences

start1 = time.time()

tagged_seq1 = Viterbi_modified_smoothing(test_tagged_words1)

end1 = time.time()

difference = end1-start1

print("Time taken in seconds: ", difference)
print(tagged_seq1)
# accuracy

check1 = [i for i, j in zip(tagged_seq1, test_run_base1) if i == j] 



accuracy1 = len(check1)/len(tagged_seq1)

accuracy1
def Viterbi_modified_RuleBasedTagger(train_set,test_set):

  # specify patterns for tagging

  # example from the NLTK book



    patterns = [ (r'.*ing$', 'VERB'),              # VERB

             (r'.*ed$', 'VERB'),               # past tense

             (r'.*es$', 'VERB'),               # 3rd singular present

             (r'.*s$', 'NOUN'),                # plural nouns

             (r'^-?[0-9]+(.[0-9]+)?$', 'NUM'), # cardinal numbers

             (r'.*', 'NOUN')             

           ]



# rule based tagger

    rule_based_tagger = nltk.RegexpTagger(patterns)



# lexicon backed up by the rule-based tagger

    lexicon_tagger = nltk.UnigramTagger(train_set, backoff=rule_based_tagger)

    a=lexicon_tagger.evaluate(test_set)

    return a

    
accu=Viterbi_modified_RuleBasedTagger(train_set,test_set)

print (accu)
## Testing

sentence_test1 = 'Android has been the best-selling OS worldwide on smartphones since 2011 and on tablets since 2013.'

words1 = word_tokenize(sentence_test1)

start = time.time()

tagged_seq1 = Viterbi(words1)

end = time.time()

difference = end-start
print(tagged_seq1)

print(difference)
## Testing

sentence_test2 = 'Android has been the best-selling OS worldwide on smartphones since 2011 and on tablets since 2013.'

words2 = word_tokenize(sentence_test2)

start = time.time()

tagged_seq2 = Viterbi_modified_smoothing(words2)

end = time.time()

difference = end-start
print(tagged_seq2)

print(difference)
## Testing



sentence_test3 = 'Android has been the best-selling OS worldwide on smartphones since 2011 and on tablets since 2013.'

words3 = word_tokenize(sentence_test3)

start = time.time()

tagged_seq3 = Viterbi(words3)

tagged_seq

end = time.time()

difference = end-start
print(tagged_seq3)

print(difference)
## Testing

sentence_test4 = 'Android is a mobile operating system developed by Google.'

words = word_tokenize(sentence_test4)

start = time.time()

tagged_seq3 = Viterbi(words)

tagged_seq3

end = time.time()

difference = end-start
print(tagged_seq3)

print(difference)
## Testing

sentence_test4 = 'Android is a mobile operating system developed by Google.'

words4 = word_tokenize(sentence_test4)

start = time.time()

tagged_seq4 = Viterbi_modified_smoothing(words4)

end = time.time()

difference = end-start
print(tagged_seq4)

print(difference)
sentence_test5 = 'Google and Twitter made a deal in 2015 that gave Google access to Twitter s firehose.'

words5 = word_tokenize(sentence_test5)

start = time.time()

tagged_seq5 = Viterbi(words5)

end = time.time()

difference = end-start
print(tagged_seq5)

print(difference)