import pandas as pd
import numpy as np
import operator
import re
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize.casual import _replace_html_entities
import csv
import os
#from textblob import TextBlob

df = pd.read_csv("../input/TestData.csv") 
df.columns = ["User","Class","Definition"]

print(df)
# Start of helper functions.

# Part of speech finder
pos = lambda tokens: pos_tag(tokens)

# Lemmatizer
lemmatize = lambda posTokens: [processPosTagsAndLemmatize(*wordPos) for wordPos in posTokens]

# Returns lemmatization based on PoS
def processPosTagsAndLemmatize(word, pos):
    return lemma.lemmatize(word, treebankToWordnetPOS(pos))

# Replaces unicode
def unicodeReplacement(tweet):
    return _replace_html_entities(tweet)

# Helper function for PoS Tagging
def treebankToWordnetPOS(treebankPosTag):
    return {'N': wordnet.NOUN}.get(treebankPosTag[0], wordnet.NOUN)

# Declares Lemmatizer
lemma = WordNetLemmatizer()

# End of helper functions
freqDict = {}
subDict = {}
sentenceDict = {}
ngramsDict = {}
linesCount = 0 
i = 1
sentence = 0
ngrams = []
n = 2 #bigrams
uniquebigrams = []

for each in df["Definition"]:
    
    linesCount = linesCount + 1
            
    sentenceDict[i] = each.count('.')
    sentence = each.count('.') + sentence
            
    text = each.lower()  # Makes each word lowercase
    text = re.sub(r'[0-9]+', '', text)  # Removes numbers
    text = re.sub(r'[^\w\s]','',text) #Punctuations
            
    #Spelling Corrector
    #text = TextBlob(text)
    #text = text.correct()
                            
    text = ' '.join([word for word in text.split() if word not in stopwords.words("english")])  # Removes stop words
    
    #bigram code starts
    input = text.split(' ')
    output = []
    for j in range(len(input)-n+1):
        output.append(input[j:j+n])

    ngramsDict[i] = [' '.join(x) for x in output]
            
    ngrams = ngrams + output
            
    tokens = word_tokenize(text)  # Tokenizes the tweets         
            
    tagged = pos(tokens)  # Grabs part of speech

    tagged = lemmatize(tagged)  # Lemmatizes
            
    subDict[i] = len(tagged)
            
    tagged = pos(tagged)  # Grabs part of speech again because it is removed in lemmatization
            
    for word in tagged:
            wordcount = 0
            if word[0] not in freqDict:  # If word is not already in the frequency list, add it
                freqDict[word[0]] = 0
                wordcount = wordcount + 1
            freqDict[word[0]] += 1  # Once word is in the frequency list, increase its frequency
                                      
    i = i + 1
    
print("Number of sentences in the corpus = " + str(sentence))  # Prints total frequency of search word

sorted_freqDict = sorted(freqDict.items(), key=operator.itemgetter(1))  # Sorts the dictionary by frequency
sorted_freqDict.reverse()  # Reverses the order
print("Number of words in the entire corpus = " + str(len(sorted_freqDict)))  # Prints total frequency of search word    
sorted_freqDictdf = pd.DataFrame(sorted_freqDict, columns=['Word', 'Frequency'])
print(sorted_freqDictdf)
sorted_subDict = sorted(subDict.items(), key=operator.itemgetter(0))
sorted_subDictdf = pd.DataFrame(sorted_subDict, columns=['User', 'Number of words'])
print(sorted_subDictdf)
sorted_sentenceDict = sorted(sentenceDict.items(), key=operator.itemgetter(0))
sorted_sentenceDictdf = pd.DataFrame(sorted_sentenceDict, columns=['User', 'Number of Sentences'])
print(sorted_sentenceDictdf)
sorted_ngramsDict = sorted(ngramsDict.items(), key=operator.itemgetter(0))
sorted_ngramsDictdf = pd.DataFrame(sorted_ngramsDict, columns=['User', 'Bigrams'])
print(sorted_ngramsDictdf)
for a in ngrams:
    if a not in uniquebigrams:
        uniquebigrams.append(a)

uniquebigramsdf = pd.DataFrame(uniquebigrams, columns=['a', 'b']) 
print(uniquebigramsdf)