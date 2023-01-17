#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import nltk
import random
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk import pos_tag
# reading the Treebank tagged sentences
nltk.download('universal_tagset')
nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))
len(nltk_data)
# Let's view the first few tagged sentences
print(nltk_data[:5])
# Splitting into train and validation sets in the ratio of 95:5
random.seed(1234)
train_set, test_set = train_test_split(nltk_data,train_size=0.95, test_size=0.05, random_state=42)

print(len(train_set))
print(len(test_set))
# Getting list of tagged words - train data
train_tagged_words = [tup for sent in train_set for tup in sent]
print(len(train_tagged_words))

# Getting list of tagged words - validtion data
test_tagged_words = [tup for sent in test_set for tup in sent]
print(len(test_tagged_words))
# tokens. i.e. the list of words from list of (word, tag) fom train set
tokens = [pair[0] for pair in train_tagged_words]
tokens[:10]
# vocabulary
V = set(tokens)
print(len(V))
# number of tags
T = sorted(list(set([pair[1] for pair in train_tagged_words])))
len(T)
# Let us view the tags
print(T)
# computing P(w/t) and storing in T x V matrix
t = len(T)
v = len(V)
w_given_t = np.zeros((t, v))
# compute Emission Probability- P(word given tag)
def word_given_tag(word, tag, train_bag = train_tagged_words):
    
    # Calculate No.of times tag t appears
    tag_list = [pair for pair in train_bag if pair[1]==tag]
    count_tag = len(tag_list)
    
    # Calculate No.of times word w has been tagged as tag t
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0]==word]
    count_w_given_tag = len(w_given_tag_list)
    
    return (count_w_given_tag, count_tag)
#Example

print("\n", "October")
print(word_given_tag('October','NOUN'))
print(word_given_tag('October','VERB'))
print(word_given_tag('October','ADJ'))

print("\n", "reported")
print(word_given_tag('reported','NOUN'))
print(word_given_tag('reported','VERB'))
print(word_given_tag('reported','ADJ'))
# Compute Transition Probability - tag2(t2) given tag1 (t1)

def t2_given_t1(t2, t1, train_bag = train_tagged_words):
    
    # Calculate No.of times tag t1 appears
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t==t1])
    
    # Calculate No.of times t1 is followed by tag t2
    count_t2_t1 = 0
    for index in range(len(tags)-1):
        if tags[index]==t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)
# examples
print(t2_given_t1('NOUN', 'ADJ'))
print(t2_given_t1('NOUN', 'DET'))
print(t2_given_t1('NOUN', 'VERB'))
print(t2_given_t1('VERB', 'NOUN'))
# Please note P(tag|start) is same as P(tag|'.')
print(t2_given_t1('DET', '.'))
print(t2_given_t1('NOUN', '.'))
print(t2_given_t1('VERB', '.'))
# creating t x t transition matrix of tags. Each column is t2, each row is t1, thus M(i, j) represents P(tj given ti)
tags_matrix = np.zeros((len(T), len(T)), dtype='float32')
for i, t1 in enumerate(list(T)):
    for j, t2 in enumerate(list(T)): 
        tags_matrix[i, j] = t2_given_t1(t2, t1)[0]/t2_given_t1(t2, t1)[1]
tags_matrix
# convert the matrix to a df for better readability
tags_df = pd.DataFrame(tags_matrix, columns = list(T), index=list(T))
# Each column is t2, each row is t1
tags_df
tags_df.loc['.', :]
# heatmap of tags matrix
plt.figure(figsize=(14, 5))
sns.heatmap(tags_df, annot=True)
plt.show()
# filter the df to get P(t2, t1) > 0.5
tags_frequent = tags_df[tags_df>0.5]
plt.figure(figsize=(14,5))
sns.heatmap(tags_frequent,annot = True)
plt.show()
# Viterbi Heuristic
def Viterbi(words, train_bag = train_tagged_words):
    state = []
    
    # Take the list of unique tags present in the corpus
    T = sorted(list(set([pair[1] for pair in train_bag])))
        
    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        p = []
        
        for tag in T:
            if key == 0: #first word has key=0
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
# Let's test our Viterbi algorithm on the validation dataset which is 5% of the entire dataset

random.seed(1234)
# list of tagged words
test_run_base = [tup for sent in test_set for tup in sent]

# take the list of words alone without the tags
test_tagged_words = [tup[0] for sent in test_set for tup in sent]
# tagging the test sentences
start = time.time()
tagged_seq = Viterbi(test_tagged_words)
end = time.time()
difference = end-start
# Below code takes around 22 mins to execute
print("Time taken in seconds: ", difference)
# accuracy
check = [i for i, j in zip(tagged_seq, test_run_base) if i == j] 
accuracy = len(check)/len(tagged_seq)
accuracy
incorrect_tagged_cases = [j for i, j in enumerate(zip(tagged_seq, test_run_base)) if j[0]!=j[1]]
incorrect_tagged_cases[:5]
# Modified Viterbi Heuristic- Approach I
def Viterbi_approach1(words, train_bag = train_tagged_words):
    state = []
    
    # Take the list of unique tags present in the corpus
    T = sorted(list(set([pair[1] for pair in train_bag])))
    V = [i[0] for i in train_bag]
    
    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        p = [] 
        
        for tag in T:
            if key == 0: #first word has key=0
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]
                
            # compute emission and state probabilities
            emission_p = word_given_tag(words[key], tag)[0]/word_given_tag(words[key], tag)[1]
            
            # modification to the original vanilla viterbi algorithm. 
            # Vocab contains the list of unique words in training dataset
            if word not in V: 
                state_probability = transition_p
            else:
                state_probability = emission_p * transition_p
                
            p.append(state_probability)
            
        pmax = max(p)
        # getting state for which probability is maximum
        state_max = T[p.index(pmax)] 
        state.append(state_max)
    return list(zip(words, state))
# tagging the test sentences
start = time.time()
transition_tagged_seq = Viterbi_approach1(test_tagged_words)
end = time.time()
difference = end-start
# Below code takes around 23 mins to execute
print("Time taken in seconds: ", difference)
# accuracy
transition_check = [i for i, j in zip(transition_tagged_seq, test_run_base) if i == j] 
transition_accuracy = len(transition_check)/len(transition_tagged_seq)
transition_accuracy
transition_incorrect_tagged_cases = [j for i, j in enumerate(zip(transition_tagged_seq, test_run_base)) if j[0]!=j[1]]
transition_incorrect_tagged_cases[:5]
# Lexicon (or unigram tagger)
unigram_tagger = nltk.UnigramTagger(train_set)
unigram_tagger.evaluate(test_set)
# patterns for tagging using a rule based tagger
patterns = [
    (r'.*\'s$', 'NOUN'),                     # possessive nouns
    (r'.*s$', 'NOUN'),                       # plural nouns
    (r'^[aA-zZ].*[0-9]+','NOUN'),            
    (r'.*ness$', 'NOUN'),                    # words ending with 'ness' such as 'sluggishness' 
    (r'.*', 'NOUN'), 
    (r'^([0-9]|[aA-zZ])+\-[aA-zZ]*$','ADJ'),     
    (r'[aA-zZ]+(ed|ing|es)$', 'VERB'),       # words ending with 'ed' or 'ing' or 'es'    
    (r'.*ly$', 'ADV'),                       # words ending with 'ly'    
    (r'^[0-9]+(.[0-9]+)?$', 'NUM'),          # cardinal numbers such as 61, 1956, 9.8, 8.45, 352.7        
    (r'(The|the|A|a|An|an)$', 'DET')
    ]
# Rule based tagger
rule_based_tagger = nltk.RegexpTagger(patterns)

# unigram tagger backed up by the rule-based tagger
rule_based_unigram_tagger = nltk.UnigramTagger(train_set, backoff = rule_based_tagger)
rule_based_unigram_tagger.evaluate(test_set)
# Bigram tagger backed up by the rule-based-unigram tagger
bigram_tagger = nltk.BigramTagger(train_set, backoff = rule_based_unigram_tagger)
bigram_tagger.evaluate(test_set)
# trigram tagger
trigram_tagger = nltk.TrigramTagger(train_set, backoff = bigram_tagger)
trigram_tagger.evaluate(test_set)
# A trigram tagger backed off by a rule based tagger.

def trigram_tagger(word, train_set = train_set):
    
    # specify patterns for tagging. I have identified most of the patterns from the first 100 sentences in universal dataset
    patterns = [
    (r'^([0-9]|[aA-zZ])+\-([0-9]|[aA-zZ])*$','ADJ'), # words such as '10-lap','30-day','York-based'
    (r'.*able$', 'ADJ'),                     # words ending with 'able' such as 'questionable'
    (r'.*ful$', 'ADJ'),                      # words ending with 'ful' such as 'useful'
    (r'.*ous$', 'ADJ'),                      # words ending with 'ous' such as 'Previous'
    
    (r'.*\'s$', 'NOUN'),                     # possessive nouns
    (r'.*s$', 'NOUN'),                       # plural nouns
    (r'^[aA-zZ].*[0-9]+','NOUN'),            # Alpha Numeric such as Door Number, Street Number etc
    (r'.*ers$', 'NOUN'),                     # words ending with 'ers' such as 'filters','workers'
    (r'.*ment$', 'NOUN'),                    # words ending with 'ment' such as 'reinvestment' 
    (r'.*town$', 'NOUN'),                    # words ending with 'town' such as 'town','downtown'  
    (r'.*ness$', 'NOUN'),                    # words ending with 'ness' such as 'sluggishness' 
    (r'^[A-Z]+([a-z]{1,2})?\.?$','NOUN'),    # words such as 'Nov.','Mr.','Inc.'
    
    (r'[aA-zZ]+(ed|ing|es)$', 'VERB'),       # words ending with 'ed' or 'ing' or 'es'    
    (r'.*ly$', 'ADV'),                       # words ending with 'ly'
    
    (r'^[0-9]+(.[0-9]+)?$', 'NUM'),          # cardinal numbers such as 61, 1956, 9.8, 8.45, 352.7        
    (r'^(0|([*|-|$].*))','X'),               # words such as '*', '0', *-1', '*T*-1', '*ICH*-1', '*?*'   
    
    (r'(The|the|A|a|An|an|That|that|This|this|Those|those|These|these)$', 'DET'), # determinants     
    (r'.*', 'NOUN')  
    ]
    
    rule_based_tagger = nltk.RegexpTagger(patterns)

    # trigram backed up by the regex tagger
    trigram_regex_tagger = nltk.TrigramTagger(train_set, backoff = rule_based_tagger)
    return trigram_regex_tagger.tag_sents([[(word)]])    
# Modified Viterbi Heuristic- Approach II - Backoff to rule based tagger in case an unknown word is encountered.
def Viterbi_approach2(words, train_bag = train_tagged_words):
    state = []
    T = sorted(list(set([pair[1] for pair in train_bag])))
    V = [i[0] for i in train_bag]
    
    # use the trigram tagger backed up by the rule based tagger for unknown words.
    for key, word in enumerate(words):
        if word not in V:
            unknown_word_tag = trigram_tagger(word)
            for sent in unknown_word_tag:
                for tup in sent:
                    state.append(tup[1])
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
# tagging the test sentences. Below code takes around 49 mins to execute
start = time.time()
viterbi_trigram_tagged_seq = Viterbi_approach2(test_tagged_words)
end = time.time()
difference = end-start

print("Time taken in seconds: ", difference)

# accuracy
viterbi_trigram_word_check = [i for i, j in zip(viterbi_trigram_tagged_seq, test_run_base) if i == j]

viterbi_trigram_accuracy = len(viterbi_trigram_word_check)/len(viterbi_trigram_tagged_seq)
viterbi_trigram_accuracy
viterbi_trigram_incorrect_tagged_cases = [j for i, j in enumerate(zip(viterbi_trigram_tagged_seq, test_run_base)) if j[0]!=j[1]]
viterbi_trigram_incorrect_tagged_cases[:5]
sentence_test1 = 'Android has been the best-selling OS worldwide on smartphones since 2011 and on tablets since 2013.'
words = word_tokenize(sentence_test1)
tagged_seq = Viterbi(words)
print(tagged_seq)
tagged_seq_modified1 = Viterbi_approach1(words)
print(tagged_seq_modified1)
tagged_seq_modified2 = Viterbi_approach2(words)
print(tagged_seq_modified2)
sentence_test2 = "Google and Twitter made a deal in 2015 that gave Google access to Twitter's firehose."
words = word_tokenize(sentence_test2)
tagged_seq = Viterbi(words)
print(tagged_seq)
tagged_seq_modified1 = Viterbi_approach1(words)
print(tagged_seq_modified1)
tagged_seq_modified2 = Viterbi_approach2(words)
print(tagged_seq_modified2)
sentence_test3 = "The 2018 FIFA World Cup is the 21st FIFA World Cup, an international football tournament contested once every four years."
words = word_tokenize(sentence_test3)
tagged_seq = Viterbi(words)
print(tagged_seq)
tagged_seq_modified1 = Viterbi_approach1(words)
print(tagged_seq_modified1)
tagged_seq_modified2 = Viterbi_approach2(words)
print(tagged_seq_modified2)