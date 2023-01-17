#Importing libraries

import nltk

import numpy as np

import pandas as pd

import random

from sklearn.model_selection import train_test_split

import pprint, time
# reading the Treebank tagged sentences

data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))
# let's check some of the tagged data

print(data[:10])
# split data into training and validation set in the ratio 95:5

random.seed(1234)

train_set, test_set = train_test_split(data, train_size=0.95, test_size=0.05)



print("Training Set Length -", len(train_set))

print("Testing Set Length -", len(test_set))

print("-" * 100)

print("Training Data -\n")

print(train_set[:10])
# Getting list of train and test tagged words

train_tagged_words = [tup for sent in train_set for tup in sent]

print("Train Tagged Words - ", len(train_tagged_words))



test_tagged_words = [tup[0] for sent in test_set for tup in sent]

print("Train Tagged Words - ", len(test_tagged_words))
# Let's have a look at the tagged words in the training set

train_tagged_words[:10]
# tokens in the train set - train_tagged_words

train_tagged_tokens = [tag[0] for tag in train_tagged_words]

train_tagged_tokens[:10]
# POS tags for the tokens in the train set - train_tagged_words



train_tagged_pos_tokens = [tag[1] for tag in train_tagged_words]

train_tagged_pos_tokens[:10]
# building the train vocabulary to a set

training_vocabulary_set = set(train_tagged_tokens)
# building the POS tags to a set

training_pos_tag_set = set(train_tagged_pos_tokens)
# let's check how many unique tags are present in training data

print(len(training_pos_tag_set))
# let's check how many words are present in vocabulary

print(len(training_vocabulary_set))
# compute emission probability for a given word for a given tag

def word_given_tag(word, tag, train_bag = train_tagged_words):

    tag_list = [pair for pair in train_bag if pair[1] == tag]

    tag_count = len(tag_list)    

    word_given_tag_list = [pair[0] for pair in tag_list if pair[0] == word]    

    word_given_tag_count = len(word_given_tag_list)    

    

    return (word_given_tag_count, tag_count)
# compute transition probabilities of a previous and next tag

def t2_given_t1(t2, t1, train_bag = train_tagged_words):

    tags = [pair[1] for pair in train_bag]

    

    t1_tags_list = [tag for tag in tags if tag == t1]

    t1_tags_count = len(t1_tags_list)

    

    t2_given_t1_list = [tags[index+1] for index in range(len(tags)-1) if tags[index] == t1 and tags[index+1] == t2]

    t2_given_t1_count = len(t2_given_t1_list)

    

    return(t2_given_t1_count, t1_tags_count)
# computing P(w/t) and storing in [Tags x Vocabulary] matrix. This is a matrix with dimension

# of len(training_pos_tag_set) X en(training_vocabulary_set)



len_pos_tags = len(training_pos_tag_set)

len_vocab = len(training_vocabulary_set)
# creating t x t transition matrix of training_pos_tag_set

# each column is t2, each row is t1

# thus M(i, j) represents P(tj given ti)



tags_matrix = np.zeros((len_pos_tags, len_pos_tags), dtype='float32')

for i, t1 in enumerate(list(training_pos_tag_set)):

    for j, t2 in enumerate(list(training_pos_tag_set)): 

        tags_matrix[i, j] = t2_given_t1(t2, t1)[0]/t2_given_t1(t2, t1)[1]
# convert the matrix to a df for better readability

tags_df = pd.DataFrame(tags_matrix, columns = list(training_pos_tag_set), index=list(training_pos_tag_set))
# Let's have a glimpse into the transition matrix

tags_df
# Importing libraries for heatmap

import matplotlib.pyplot as plt

import seaborn as sns
# heatmap of tags matrix

# T(i, j) means P(tag j given tag i)

plt.figure(figsize=(14, 8))

sns.heatmap(tags_df, annot = True)

plt.show()
# frequent tags

# filter the df to get P(t2, t1) > 0.5

tags_frequent = tags_df[tags_df>0.5]

plt.figure(figsize=(14, 8))

sns.heatmap(tags_frequent, annot = True)

plt.show()
# Vanilla Viterbi Algorithm

def Vanilla_Viterbi(words, train_bag = train_tagged_words):

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

            emission_p = word_given_tag(words[key], tag)[0]/word_given_tag(words[key], tag)[1]

            state_probability = emission_p * transition_p    

            p.append(state_probability)

            

        pmax = max(p)

        # getting state for which probability is maximum

        state_max = T[p.index(pmax)] 

        state.append(state_max)

    return list(zip(words, state))
# Let's test our Viterbi algorithm on a few sample sentences of test dataset



random.seed(1234)



# choose random 5 sents

rndom = [random.randint(1, len(test_set)) for x in range(5)]



# list of sents

test_run = [test_set[i] for i in rndom]



# list of tagged words

test_run_base = [tup for sent in test_run for tup in sent]



# list of untagged words

test_tagged_words = [tup[0] for sent in test_run for tup in sent]
# tagging the test sentences

start = time.time()

tagged_seq = Vanilla_Viterbi(test_tagged_words)

end = time.time()

difference = end-start



print("Time taken in seconds: ", difference)



# accuracy

vanilla_viterbi_word_check = [i for i, j in zip(tagged_seq, test_run_base) if i == j] 

vanilla_viterbi_accuracy = len(vanilla_viterbi_word_check)/len(tagged_seq) * 100

print('Vanilla Viterbi Algorithm Accuracy: ', vanilla_viterbi_accuracy)
# let's check the incorrectly tagged words

incorrect_tagged_words = [j for i, j in enumerate(zip(tagged_seq, test_run_base)) if j[0] != j[1]]



print("Total Incorrect Tagged Words :", len(incorrect_tagged_words))

print("\n")

print("Incorrect Tagged Words :", incorrect_tagged_words)
# Unknown words 



test_vocabulary_set = set([t for t in test_tagged_words])



unknown_words = list(test_vocabulary_set - training_vocabulary_set)

print("Total Unknown words :", len(unknown_words))

print("\n")

print("Unknown Words :", unknown_words)
# Lexicon (or unigram tagger)



unigram_tagger = nltk.UnigramTagger(train_set)

accuracy_unigram_tagger = unigram_tagger.evaluate(test_set)

print("The accuracy of the Unigram Tagger is -", accuracy_unigram_tagger)
# patterns for tagging using a rule based regex tagger -



patterns = [

    (r'^[aA-zZ].*[0-9]+','NOUN'),  # Alpha Numeric

    (r'.*ness$', 'NOUN'),

    (r'.*\'s$', 'NOUN'),              # possessive nouns

    (r'.*s$', 'NOUN'),                # plural nouns

    (r'.*', 'NOUN'),    

    (r'.*ly$', 'ADV'),

    (r'^(0|([*|-|$].*))','X'), # Any special character combination

    (r'.*ould$', 'X'), # modals

    (r'(The|the|A|a|An|an)$', 'DET'),

    (r'^([0-9]|[aA-zZ])+\-[aA-zZ]*$','ADJ'),

    (r'.*able$', 'ADJ'), # adjective like 100-megabytes 237-Seats

    (r'[aA-zZ]+(ed|ing|es)$', 'VERB'), # Any word ending with 'ing' or 'ed' is a verb

    (r'[0-9].?[,\/]?[0-9]*','NUM')# Numbers 

    ]
# rule based tagger



rule_based_tagger = nltk.RegexpTagger(patterns)



# unigram tagger backed up by the rule-based tagger

rule_based_unigram_tagger = nltk.UnigramTagger(train_set, backoff = rule_based_tagger)



accuracy_rule_based_unigram_tagger = rule_based_unigram_tagger.evaluate(test_set)



print("The accuracy of the Unigram Tagger backed up by the RegexpTagger is -", accuracy_rule_based_unigram_tagger)
# Bigram tagger



bigram_tagger = nltk.BigramTagger(train_set, backoff=rule_based_unigram_tagger)

bigram_tagger.evaluate(test_set)

accuracy_bigram_tagger = bigram_tagger.evaluate(test_set)

print(accuracy_bigram_tagger)
# Trigram tagger



trigram_tagger = nltk.TrigramTagger(train_set, backoff = bigram_tagger)

trigram_tagger.evaluate(test_set)

accuracy_trigram_tagger = trigram_tagger.evaluate(test_set)

print("The accuracy of the Trigram Tagger backed up by the bigram_tagger is -", accuracy_trigram_tagger)
# use transition probability of tags when emission probability is zero (in case of unknown words)



def Vanilla_Viterbi_for_Unknown_Words(words, train_bag = train_tagged_words):

    state = []

    T = list(set([pair[1] for pair in train_bag]))

    

    for key, word in enumerate(words):

        #initialise list of probability column for a given observation

        p = [] 

        p_transition =[] # list for storing transition probabilities

        for tag in T:

            if key == 0:

                transition_p = tags_df.loc['.', tag]

            else:

                transition_p = tags_df.loc[state[-1], tag]

                

            # compute emission and state probabilities

            emission_p = word_given_tag(words[key], tag)[0]/word_given_tag(words[key], tag)[1]

            state_probability = emission_p * transition_p    

            p.append(state_probability)

            p_transition.append(transition_p)

            

        pmax = max(p)

        state_max = T[p.index(pmax)] 

        

      

        # if probability is zero (unknown word) then use transition probability

        if(pmax==0):

            pmax = max(p_transition)

            state_max = T[p_transition.index(pmax)]

                           

        else:

            state_max = T[p.index(pmax)] 

        

        state.append(state_max)

    return list(zip(words, state))
# tagging the test sentences

start = time.time()

tagged_seq = Vanilla_Viterbi_for_Unknown_Words(test_tagged_words)

end = time.time()

difference = end-start



print("Time taken in seconds: ", difference)



# accuracy

check = [i for i, j in zip(tagged_seq, test_run_base) if i == j] 

accuracy = len(check)/len(tagged_seq)

print('Vanilla Viterbi for Unknown Words Accuracy: ',accuracy*100)
# lets create a list containing tuples of POS tags and POS tag occurance probability, based on training data

tag_prob = []

total_tag = len([tag for word,tag in train_tagged_words])

for t in training_pos_tag_set:

    each_tag = [tag for word,tag in train_tagged_words if tag==t]

    tag_prob.append((t,len(each_tag)/total_tag))



tag_prob
# use transition probability of tags when emission probability is zero (in case of unknown words)



def Vanilla_Viterbi_for_Unknown_Words_Modified(words, train_bag = train_tagged_words):

    state = []

    T = list(set([pair[1] for pair in train_bag]))

    

    for key, word in enumerate(words):

        #initialise list of probability column for a given observation

        p = [] 

        p_transition =[] # list for storing transition probabilities

       

        for tag in T:

            if key == 0:

                transition_p = tags_df.loc['.', tag]

            else:

                transition_p = tags_df.loc[state[-1], tag]

                

            # compute emission and state probabilities

            emission_p = word_given_tag(words[key], tag)[0]/word_given_tag(words[key], tag)[1]

            state_probability = emission_p * transition_p    

            p.append(state_probability)

            

            # find POS tag occurance probability

            tag_p = [pair[1] for pair in tag_prob if pair[0]==tag ]

            

            # calculate the transition prob weighted by tag occurance probability.

            transition_p = tag_p[0]*transition_p             

            p_transition.append(transition_p)

            

        pmax = max(p)

        state_max = T[p.index(pmax)] 

        

      

        # if probability is zero (unknown word) then use weighted transition probability

        if(pmax==0):

            pmax = max(p_transition)

            state_max = T[p_transition.index(pmax)]                 

                           

        else:

            state_max = T[p.index(pmax)] 

        

        state.append(state_max)

    return list(zip(words, state))
# tagging the test sentences

start = time.time()

tagged_seq = Vanilla_Viterbi_for_Unknown_Words_Modified(test_tagged_words)

end = time.time()

difference = end-start



print("Time taken in seconds: ", difference)



# accuracy

viterbi_word_check = [i for i, j in zip(tagged_seq, test_run_base) if i == j] 

accuracy_viterbi_modified = len(viterbi_word_check)/len(tagged_seq) * 100

print('Modified Vanilla Viterbi for Unknown Words Accuracy: ', accuracy_viterbi_modified)
# A trigram tagger backed off by a rule based tagger.



def trigram_tagger(word, train_set = train_set):

    

    patterns = [

    (r'[aA-zZ]+(ed|ing|es)$', 'VERB'), # Any word ending with 'ing' or 'ed' is a verb



    (r'.*ly$', 'ADV'),

        

    (r'^([0-9]|[aA-zZ])+\-[aA-zZ]*$','ADJ'),

    (r'.*able$', 'ADJ'), 

    (r'.*ful$', 'ADJ'),

    (r'.*ous$', 'ADJ'),

        

    (r'^[aA-zZ].*[0-9]+','NOUN'),     # Alpha Numeric

    (r'.*ness$', 'NOUN'),

    (r'.*\'s$', 'NOUN'),              # possessive nouns - words ending with 's

    (r'.*s$', 'NOUN'),                # plural nouns

    (r'.*ers$', 'NOUN'),              # eg.- kinderganteners, autobioghapgers

    (r'.*ment$', 'NOUN'),

    (r'.*town$', 'NOUN'),

        

    (r'^(0|([*|-|$].*))','X'), # Any special character combination

    (r'.*ould$', 'X'),

        

    (r'(The|the|A|a|An|an|That|that|This|this|Those|those|These|these)$', 'DET'), # That/this/these/those belong to the category of Demonstrative determiners

    (r'[0-9].?[,\/]?[0-9]*','NUM'), # Numbers 

        

    (r'.*', 'NOUN')

    ]



    regex_based_tagger = nltk.RegexpTagger(patterns)



    # trigram backed up by the regex tagger

    trigram_regex_tagger = nltk.TrigramTagger(train_set, backoff = regex_based_tagger)

    return trigram_regex_tagger.tag_sents([[(word)]])
# viterbi with handling for unknown words from regex tagger



def Viterbi_Trigram_Tagger(words, train_bag = train_tagged_words):

    state = []

    T = list(set([pair[1] for pair in train_bag]))

    

    for key, word in enumerate(words):

        # unknown words from trigram taggr

        if word not in training_vocabulary_set:

            unk_word_tag=trigram_tagger(word)

            for sent in unk_word_tag:

                for tup in sent:

                    state.append(tup[1])

        # rest remains same            

        else:            

            p = [] 

            for tag in T:

                if key == 0:

                    transition_p = tags_df.loc['.', tag]

                else:

                    transition_p = tags_df.loc[state[-1], tag]

                

            # compute emission and state probabilities

                emission_p = word_given_tag(words[key], tag)[0]/word_given_tag(words[key], tag)[1]

                state_probability = emission_p * transition_p    

                p.append(state_probability)

            

            pmax = max(p)

            # getting state for which probability is maximum

            state_max = T[p.index(pmax)] 

            state.append(state_max)

            

    return list(zip(words, state))
# tagging the test sentences

start = time.time()

tagged_seq = Viterbi_Trigram_Tagger(test_tagged_words)

end = time.time()

difference = end-start



print("Time taken in seconds: ", difference)



# accuracy

viterbi_trigram_word_check = [i for i, j in zip(tagged_seq, test_run_base) if i == j] 

viterbi_trigram_accuracy = len(viterbi_trigram_word_check)/len(tagged_seq) * 100

print('Modified Viterbi Trigram Tagger Accuracy: ', viterbi_trigram_accuracy)
acccuracy_data = [['Vanilla Viterbi', vanilla_viterbi_accuracy], 

                  ['Vanilla Viterbi Modified', accuracy_viterbi_modified], 

                  ['Unigram Tagger', accuracy_unigram_tagger * 100],

                  ['Unigram + RegexpTagger', accuracy_rule_based_unigram_tagger * 100],

                  ['Bigram Tagger + Unigram_tagger', accuracy_bigram_tagger*100],

                  ['Trigram Tagger + Bigram_tagger', accuracy_trigram_tagger*100],

                  ['Viterbi + Trigram_tagger', viterbi_trigram_accuracy]]



acccuracy_data_df = pd.DataFrame(acccuracy_data, columns = ['Tagging_Algorithm', 'Tagging_Accuracy'])



acccuracy_data_df.set_index('Tagging_Algorithm', drop = True, inplace = True)



acccuracy_data_df
acccuracy_data_df.plot.line(rot = 90, legend = False)
from nltk.tokenize import word_tokenize
## Testing

sentence_test_1 = 'Google and Twitter made a deal in 2015 that gave Google access to Twitter\'s firehose.'

words = word_tokenize(sentence_test_1)

tagged_seq = Vanilla_Viterbi(words)

print(tagged_seq)
tagged_seq_modified = Viterbi_Trigram_Tagger(words)

print(tagged_seq_modified)
sentence_test_2='Android has been the best-selling OS worldwide on smartphones since 2011 and on tablets since 2013.'

words = word_tokenize(sentence_test_2)

tagged_seq = Vanilla_Viterbi(words)

print(tagged_seq)
tagged_seq_modified = Viterbi_Trigram_Tagger(words)

print(tagged_seq_modified)
sentence_test_3='I Instagrammed a Facebook post taken from Android smartphone and uploaded results to Youtube.'

words = word_tokenize(sentence_test_3)

tagged_seq = Vanilla_Viterbi(words)

print(tagged_seq)
tagged_seq_modified = Viterbi_Trigram_Tagger(words)

print(tagged_seq_modified)