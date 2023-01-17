import warnings 

warnings.filterwarnings('ignore')

import nltk 

import pandas as pd 

from nltk.tokenize import RegexpTokenizer #will use this to remove puntuation and tokenize the text

import matplotlib.pyplot as plt

%matplotlib inline



df = pd.read_csv("../input/train.csv")
df_eap = df[df['author'] == 'EAP'] #separate the 3 authors

df_hpl = df[df['author'] == 'HPL']

df_mws = df[df['author'] == 'MWS']



text_eap = df_eap['text'].str.cat(sep = ' ').lower() #concatenate all sentences for each author and set all characters to lower case

text_hpl = df_hpl['text'].str.cat(sep = ' ').lower()

text_mws = df_mws['text'].str.cat(sep = ' ').lower()



#I am changing everything to lower case because I will do some word frequency analysis, 

#I don't want things like 'Old' and 'old counted as different words



tokenizer = RegexpTokenizer(r'\w+') 

tokens_eap = tokenizer.tokenize(text_eap) #this will return a list of tokens(words) in lower case with punctuation removed

tokens_hpl = tokenizer.tokenize(text_hpl)

tokens_mws = tokenizer.tokenize(text_mws)

#note: one problem with the above approach is phrases like the "man's hat" will be tokenized into 3 tokens (man, s, hat)

pos_list_eap = nltk.pos_tag(tokens_eap) #this step will add the pos tags

pos_list_hpl = nltk.pos_tag(tokens_hpl)

pos_list_mws = nltk.pos_tag(tokens_mws)
# function to test if something is an adjective, comparitive or superlatives

def is_adj(pos):

    result = False

    if pos in ('JJ','JJR','JJS'):

        result = True

    return result



adj_eap = [word for word, pos in pos_list_eap if is_adj(pos) and len(word) > 1] #this is just a list of all adjectives for EAP

adj_hpl = [word for word, pos in pos_list_hpl if is_adj(pos) and len(word) > 1]

adj_mws = [word for word, pos in pos_list_mws if is_adj(pos) and len(word) > 1]



# I added the > 1 test because 'i' is sometimes marked as JJ. I think this is because JJ also covers numerals and ordinals, maybe 

#'i' is being seen as Roman numeral for one(?)



freq_eap = nltk.FreqDist(adj_eap) #this gets the frequency distribution for the adjectives in the list adj_eap

freq_hpl = nltk.FreqDist(adj_hpl)

freq_mws = nltk.FreqDist(adj_mws)



#if you want to print out the top twenty list use:

#print(freq_eap.most_common(20))

#print(freq_hpl.most_common(20))

#print(freq_mws.most_common(20))
eap = len(adj_eap)/len(tokens_eap)#number of adjectives divided by total number of words for EAP

hpl = len(adj_hpl)/len(tokens_hpl)

mws = len(adj_mws)/len(tokens_mws)



d = {'EAP':eap, 'HPL': hpl, 'MWS':mws} 



plt.bar(range(len(d)), d.values(), align='center')

plt.xticks(range(len(d)), d.keys())

plt.title("Adjectives used as a fraction of total words for each author")



plt.show()
freq_eap.plot(20,cumulative=False,title='top 20 adjectives for EAP') #looking just at top 20 adjectives  for EAP
freq_hpl.plot(20,cumulative=False,title='top 20 adjectives for HPL')  # HPL
freq_mws.plot(20,cumulative=False, title='top 20 adjectives for MWS')  # MWS
from math import*

 

def jaccard_similarity(x,y):

    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))

    union_cardinality = len(set.union(*[set(x), set(y)]))

    return intersection_cardinality/float(union_cardinality)



set_eap = set(adj_eap)

set_hpl = set(adj_hpl)

set_mws = set(adj_mws)

print('Jaccard Similarity scores for adjectives used by each pair of authors')

print('EAP - HPL: ' + str(jaccard_similarity(set_eap,set_hpl)))

print('EAP - MWS: ' + str(jaccard_similarity(set_eap,set_mws)))

print('MWS - HPL: ' + str(jaccard_similarity(set_hpl,set_mws)))
def is_verb(pos):

    result = False

    if pos in ('VB','VBD','VBG','VBN','VBP','VBZ'):

        result = True

    return result



verb_eap = [word for word, pos in pos_list_eap if is_verb(pos) and len(word) > 1] 

verb_hpl = [word for word, pos in pos_list_hpl if is_verb(pos) and len(word) > 1]

verb_mws = [word for word, pos in pos_list_mws if is_verb(pos) and len(word) > 1]

#the >1 test prevents things like 'i' being tagged, there are no verbs in English with just 1 character

freq_eap_verb = nltk.FreqDist(verb_eap) 

freq_hpl_verb = nltk.FreqDist(verb_hpl)

freq_mws_verb = nltk.FreqDist(verb_mws)



#if you want to print out the top twenty list use:

#print(freq_eap_verb.most_common(20))

#print(freq_hpl_verb.most_common(20))

#print(freq_mws_verb.most_common(20))
eap = len(verb_eap)/len(tokens_eap)

hpl = len(verb_hpl)/len(tokens_hpl)

mws = len(verb_mws)/len(tokens_mws)



d = {'EAP':eap, 'HPL': hpl, 'MWS':mws} 



plt.bar(range(len(d)), d.values(), align='center')

plt.xticks(range(len(d)), d.keys())

plt.title("Verbs used as a fraction of total words for each author")

plt.show()
freq_eap_verb.plot(20,cumulative=False,title='top 20 verbs for EAP')
freq_hpl_verb.plot(20,cumulative=False,title='top 20 verbs for HPL')
freq_mws_verb.plot(20,cumulative=False,title='top 20 verbs for MWS')
from math import*

 

def jaccard_similarity(x,y):

    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))

    union_cardinality = len(set.union(*[set(x), set(y)]))

    return intersection_cardinality/float(union_cardinality)



set_eap = set(verb_eap)

set_hpl = set(verb_hpl)

set_mws = set(verb_mws)

 

print('Jaccard Similarity scores for verbs used by each pair of authors')

print('EAP - HPL: ' + str(jaccard_similarity(set_eap,set_hpl)))

print('EAP - MWS: ' + str(jaccard_similarity(set_eap,set_mws)))

print('MWS - HPL: ' + str(jaccard_similarity(set_hpl,set_mws)))
def is_noun(pos):

    result = False

    if pos in ('NN','NNP','NNPS','NNS'):

        result = True

    return result



noun_eap = [word for word, pos in pos_list_eap if is_noun(pos) and len(word) > 1]

noun_hpl = [word for word, pos in pos_list_hpl if is_noun(pos) and len(word) > 1]

noun_mws = [word for word, pos in pos_list_mws if is_noun(pos) and len(word) > 1]

#the >1 test gets rid of the 's' problem mentioned above, it also gets rid of the pronoun 'i' which dominates

#the plots

freq_eap_noun = nltk.FreqDist(noun_eap) 

freq_hpl_noun = nltk.FreqDist(noun_hpl)

freq_mws_noun = nltk.FreqDist(noun_mws)



#if you want to print out the top twenty list use:

#print(freq_eap_noun.most_common(20))

#print(freq_hpl_noun.most_common(20))

#print(freq_mws_noun.most_common(20))
eap = len(noun_eap)/len(tokens_eap)

hpl = len(noun_hpl)/len(tokens_hpl)

mws = len(noun_mws)/len(tokens_mws)



d = {'EAP':eap, 'HPL': hpl, 'MWS':mws} 



plt.bar(range(len(d)), d.values(), align='center')

plt.xticks(range(len(d)), d.keys())

plt.title("Nouns used as a fraction of total words for each author")

plt.show()
freq_eap_noun.plot(20,cumulative=False,title='top 20 nouns for EAP')
freq_hpl_noun.plot(20,cumulative=False,title='top 20 nouns for HPL')
freq_mws_noun.plot(20,cumulative=False,title='top 20 nouns for MWS')
from math import*

 

def jaccard_similarity(x,y):

    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))

    union_cardinality = len(set.union(*[set(x), set(y)]))

    return intersection_cardinality/float(union_cardinality)



set_eap = set(noun_eap)

set_hpl = set(noun_hpl)

set_mws = set(noun_mws)

 

print('Jaccard Similarity scores for nouns used by each pair of authors')

print('EAP - HPL: ' + str(jaccard_similarity(set_eap,set_hpl)))

print('EAP - MWS: ' + str(jaccard_similarity(set_eap,set_mws)))

print('MWS - HPL: ' + str(jaccard_similarity(set_hpl,set_mws)))
stopwords = nltk.corpus.stopwords.words('english')



eap_stop = [t for t in tokens_eap if t in stopwords]

hpl_stop = [t for t in tokens_hpl if t in stopwords]

mws_stop = [t for t in tokens_mws if t in stopwords]



freq_eap_stop = nltk.FreqDist(eap_stop) 

freq_hpl_stop = nltk.FreqDist(hpl_stop)

freq_mws_stop = nltk.FreqDist(mws_stop)



freq_eap_stop.plot(20,cumulative=False,title='top 20 stop words for EAP')
freq_hpl_stop.plot(20,cumulative=False,title='top 20 stop words for HPL')
freq_mws_stop.plot(20,cumulative=False,title='top 20 stop words for MWS')
from math import*

 

def jaccard_similarity(x,y):

    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))

    union_cardinality = len(set.union(*[set(x), set(y)]))

    return intersection_cardinality/float(union_cardinality)



set_eap = set(eap_stop)

set_hpl = set(hpl_stop)

set_mws = set(mws_stop)

 

print('Jaccard Similarity scores for nouns used by each pair of authors')

print('EAP - HPL: ' + str(jaccard_similarity(set_eap,set_hpl)))

print('EAP - MWS: ' + str(jaccard_similarity(set_eap,set_mws)))

print('MWS - HPL: ' + str(jaccard_similarity(set_hpl,set_mws)))
pronoun = ['he','she','him','her','his']



eap_pronoun = [t for t in tokens_eap if t in pronoun]

hpl_pronoun = [t for t in tokens_hpl if t in pronoun]

mws_pronoun = [t for t in tokens_mws if t in pronoun]



freq_eap_pronoun = nltk.FreqDist(eap_pronoun) 

freq_hpl_pronoun = nltk.FreqDist(hpl_pronoun)

freq_mws_pronoun = nltk.FreqDist(mws_pronoun)
freq_eap_pronoun.plot(5,cumulative=False,title='top third person pronouns for EAP')
freq_hpl_pronoun.plot(5,cumulative=False,title='top third person pronouns for HPL')
freq_mws_pronoun.plot(5,cumulative=False,title='top third person pronouns for MWS')
eap = len(eap_stop)/len(tokens_eap)

hpl = len(hpl_stop)/len(tokens_hpl)

mws = len(mws_stop)/len(tokens_mws)



d = {'EAP':eap, 'HPL': hpl, 'MWS':mws} 



plt.bar(range(len(d)), d.values(), align='center')

plt.xticks(range(len(d)), d.keys())

plt.title("Stopwords as a fraction of total words for each author")

plt.show()