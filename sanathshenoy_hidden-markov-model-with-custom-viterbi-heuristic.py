#Importing libraries
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

import seaborn as sns
nltk.download('treebank')
nltk.download('universal_tagset')
nltk.download('punkt')
import time
import random
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.tag import DefaultTagger,UnigramTagger,BigramTagger
#suppressing unecessary warnings
warnings.simplefilter(action='ignore')
# reading the Treebank tagged sentences using universal tagset
nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))
nltk_data[:10]
random.seed(100)
train, test = train_test_split(nltk_data,test_size=0.05)

print(len(train))
print(len(test))

# Getting list of tagged words for training 
taggedwords_train = [tup for sent in train for tup in sent]
len(taggedwords_train)
word_list = [pair[0] for pair in taggedwords_train]
word_list[:10]
tag_list = [pair[1] for pair in taggedwords_train]
# unique set of words
vocab = set(word_list)
print(len(vocab))
# unique set of tags
u_tags = set(tag_list)
print(len(u_tags))
u_tags
# compute word given tag ie the emmission probability for viterbi algorithm
def compute_wordgiven_tag(word, tag, train_bag = taggedwords_train):
    taglist = [pair for pair in train_bag if pair[1]==tag]
    count_of_tags = len(taglist)
    wordgiven_taglist = [pair[0] for pair in taglist if pair[0]==word]
    countwordgiven_taglist = len(wordgiven_taglist) 
    return (countwordgiven_taglist, count_of_tags)
#compute tag2(t2) given tag1 (t1), i.e. Transition Probability
def compute_t2_given_t1(t2, t1, train_bag = taggedwords_train):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t==t1])
    count_t2_t1 = 0
    for index in range(len(tags)-1):
        if tags[index]==t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)
def compute_tagmatrix():
    #matrix of transition probabilities
    tags_matrix = np.zeros((len(u_tags), len(u_tags)), dtype='float32')
    for i, t1 in enumerate(list(u_tags)):
        for j, t2 in enumerate(list(u_tags)):
            tags_matrix[i, j] = compute_t2_given_t1(t2, t1)[0]/compute_t2_given_t1(t2, t1)[1]
    return tags_matrix  
# Viterbi Heuristic as explained in the session of syntactic analysis
def veterbi(words, train_bag = taggedwords_train):
    state = []
    T = list(set([pair[1] for pair in train_bag]))
    
    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        p = [] 
        for tag in T:
            if key == 0:
                transition_p = tag_probabilities.loc['.', tag]
            else:
                transition_p = tag_probabilities.loc[state[-1], tag]
                
            # compute emission and state probability
            emission_p = compute_wordgiven_tag(words[key], tag)[0]/compute_wordgiven_tag(words[key], tag)[1]
            state_probability = emission_p * transition_p    
            p.append(state_probability)
            
        pmax = max(p)
        # getting state for which probability is maximum
        state_max = T[p.index(pmax)] 
        state.append(state_max)
    return list(zip(words, state))
pattern_string = [
        
        (r'[A-Z ]+', 'NOUN'),  # words with capital letters as abreviations
        (r'[-+]?\d*\.\d+|\d+', 'NUM'), #floating point numbers
        (r'.*ing$', 'VERB'),
        (r'.*ed$', 'VERB'),  # actions with ed
        (r'.*es$', 'VERB'), #action ending with es
       
]
def rule_based_tagging(word, pattern_string):
    regex_tagger = nltk.RegexpTagger(pattern_string)
    wordtag = regex_tagger.tag(nltk.word_tokenize(word))
    for tag in wordtag:
        return tag[1]
def rulebased_veterbi(words, train_bag = taggedwords_train):
    state = []
    T = list(set([pair[1] for pair in train_bag]))
    
    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        p = [] 
        for tag in T:
            if key == 0:
                transition_p = tag_probabilities.loc['.', tag]
            else:
                transition_p = tag_probabilities.loc[state[-1], tag]
                
            # compute emission and state probability
            emission_p = compute_wordgiven_tag(words[key], tag)[0]/compute_wordgiven_tag(words[key], tag)[1]
            state_probability = emission_p * transition_p    
            p.append(state_probability)
            
        pmax = max(p)
        if pmax == 0.0:       
            state_max = rule_based_tagging(word, pattern_string)
            if state_max is None :
                state_max = "."
        else :
            state_max = T[p.index(pmax)]  
        state.append(state_max)
    return list(zip(words, state))
def bigram_tagger(wording) :
    t0 = DefaultTagger('NOUN')
    t1 = UnigramTagger(train, backoff=t0)
    t2 = BigramTagger(train, backoff=t1)
    listvalue=t2.tag(nltk.word_tokenize(wording))
    return listvalue[0][1]
def bigram_veterbi(words, train_bag = taggedwords_train):
    state = []
    T = list(set([pair[1] for pair in train_bag]))
    
    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        p = [] 
        for tag in T:
            if key == 0:
                transition_p = tag_probabilities.loc['.', tag]
            else:
                transition_p = tag_probabilities.loc[state[-1], tag]
                
            # compute emission and state probability
            emission_p = compute_wordgiven_tag(words[key], tag)[0]/compute_wordgiven_tag(words[key], tag)[1]
            state_probability = emission_p * transition_p    
            p.append(state_probability)
            
        pmax = max(p)
        if pmax == 0.0:       
            state_max = bigram_tagger(word)
            if state_max is None :
                state_max = "."
        else :
            state_max = T[p.index(pmax)]  
        state.append(state_max)
    return list(zip(words, state))

#test veterbi all three types
def run_veterbi(choice,content):
    start = time.time()
    if(choice=="Veterbi"):
        print("running plain Veterbi")
        tagged_seq = veterbi(content,taggedwords_train)
    elif(choice=="Rule Based Veterbi Extension"):
        print("Rule Based veterbi Extension")
        tagged_seq = rulebased_veterbi(content,taggedwords_train)
    elif(choice=="Bigram Based Veterbi Extension"):
        print("Bigram Based veterbi Extension")
        tagged_seq = bigram_veterbi(content,taggedwords_train)
    end = time.time()
    difference = end-start
    total_time = str(difference)
    return tagged_seq,total_time
def generate_heatmap(data,threshold):
    if(threshold is None):
        plot_data = tag_probabilities
    else:
        plot_data = tag_probabilities [tag_probabilities>threshold]
    plt.figure(figsize=(10, 10))
    sns.heatmap(plot_data, annot=True, cmap="YlGnBu")
    plt.show()
def get_unseen_filetestdata():
    sentences = "I Instagrammed a Facebook post taken from Android smartphone and uploaded results to Youtube."
    words = word_tokenize(sentences)
    return words
metric_map= {}
# get the incorrect cases and accuracy based on the examples elaborated from syntax analysis
def compute_summary(result,name,log):
    info_map={}
    check = [i for i, j in zip(result, test_run_base) if i == j] 
    accuracy = len(check)/len(result)
    miss_tagged_content=[[test_run_base[i-1],j] for i, j in enumerate(zip(result, test_run_base)) if j[0]!=j[1]]
    if(log==True):
        info_map["Accuracy"]=accuracy
        metric_map[name]=info_map
    return miss_tagged_content,accuracy
def get_evaluation_summary():
    metric_frame =pd.DataFrame(metric_map)
    metric_frame.reset_index()
    return metric_frame

tag_probabilities=compute_tagmatrix()
tag_probabilities = pd.DataFrame(tag_probabilities, columns = list(u_tags), index=list(u_tags))
tag_probabilities
generate_heatmap(tag_probabilities,None)
generate_heatmap(tag_probabilities,0.5)
random.seed(100)
test_run_base = [tup for sent in test for tup in sent]
test_tagged_words = [tup[0] for sent in test for tup in sent]
print("length of test set word: " + str(len(test_tagged_words)))
tagged_sequence,value=run_veterbi("Veterbi",test_tagged_words[:100]) #choose small size training data
tagged_sequence
misstagged_content,accuracy=compute_summary(tagged_sequence,"Veterbi",True)
accuracy 
len(misstagged_content)
unseen_words= get_unseen_filetestdata()
print(len(unseen_words))
tagged_sequence,value=run_veterbi("Veterbi",unseen_words)
tagged_sequence
tagged_sequence,completion_time=run_veterbi("Rule Based Veterbi Extension",test_tagged_words[:100])
tagged_sequence
misstagged_content,accuracy=compute_summary(tagged_sequence,"Rule Based Veterbi Extension",True)
accuracy
len(misstagged_content)
tagged_sequence,value=run_veterbi("Rule Based Veterbi Extension",unseen_words)
tagged_sequence
tagged_sequence,completion_time=run_veterbi("Bigram Based Veterbi Extension",test_tagged_words[:100])
misstagged_content,accuracy=compute_summary(tagged_sequence,"Bigram Based Veterbi Extension",True)
accuracy
tagged_sequence
len(misstagged_content)
tagged_sequence,value=run_veterbi("Bigram Based Veterbi Extension",unseen_words)
tagged_sequence
metric_df=get_evaluation_summary()
metric_df.T